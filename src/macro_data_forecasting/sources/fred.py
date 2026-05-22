"""FRED/ALFRED source client for point-in-time macro observations."""

import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from macro_data_forecasting.config import get_settings
from macro_data_forecasting.database import (
    REQUIRED_OBSERVATION_COLUMNS,
    upsert_observations,
)

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
NORMALIZED_COLUMNS = list(REQUIRED_OBSERVATION_COLUMNS)
VALID_OUTPUT_TYPES = {1, 2, 3, 4}
VALID_VINTAGE_MODES = {"current", "initial_release", "realtime_period"}


class FredApiError(RuntimeError):
    """Raised when the FRED API request or response is invalid."""


class FredClient:
    """Client for fetching normalized FRED/ALFRED observations."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = FRED_API_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        """Initialize a FRED client with settings-backed credentials."""
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.fred_api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _request_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            msg = "FRED_API_KEY is required for FRED requests."
            raise FredApiError(msg)

        request_params = {
            key: value for key, value in params.items() if value is not None
        }
        request_params["api_key"] = self.api_key
        request_params["file_type"] = "json"
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=request_params,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff_seconds * attempt)
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff_seconds * attempt)
                continue

            if response.status_code >= 400:
                msg = (
                    "FRED API request failed with status "
                    f"{response.status_code}: {response.text}"
                )
                raise FredApiError(msg)

            try:
                payload = response.json()
            except ValueError as exc:
                msg = "FRED API response was not valid JSON."
                raise FredApiError(msg) from exc

            if "error_code" in payload or "error_message" in payload:
                msg = (
                    "FRED API returned an error: "
                    f"{payload.get('error_code')} {payload.get('error_message')}"
                )
                raise FredApiError(msg)

            return payload

        if last_error is not None:
            msg = "FRED API request failed after retries."
            raise FredApiError(msg) from last_error
        msg = "FRED API request failed after retries due to transient HTTP errors."
        raise FredApiError(msg)

    def fetch_series_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        frequency: str | None = None,
        aggregation_method: str | None = None,
        output_type: int | None = None,
        vintage_mode: str = "current",
    ) -> pd.DataFrame:
        """Fetch and normalize observations for one FRED/ALFRED series."""
        request_output_type = _resolve_output_type(output_type, vintage_mode)
        payload = self._request_json(
            "series/observations",
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
                "frequency": frequency,
                "aggregation_method": aggregation_method,
                "output_type": request_output_type,
            },
        )
        observations = payload.get("observations")
        if observations is None:
            msg = "FRED API response did not include an observations field."
            raise FredApiError(msg)
        if not observations:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)

        frame = pd.DataFrame(observations)
        fetched_at = pd.Timestamp.now(tz="UTC")
        values = frame["value"].replace({".": np.nan, "": np.nan})
        release_dates = _extract_release_dates(
            frame=frame,
            payload=payload,
            realtime_start=realtime_start,
            vintage_mode=vintage_mode,
        )

        normalized = pd.DataFrame(
            {
                "series_id": series_id,
                "date": pd.to_datetime(frame["date"], errors="raise").dt.date,
                "value": pd.to_numeric(values, errors="coerce"),
                "source": "fred",
                "release_date": release_dates,
                "fetched_at": fetched_at,
            },
            columns=NORMALIZED_COLUMNS,
        )
        return normalized

    def fetch(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """Fetch normalized FRED observations for a source-client workflow."""
        return self.fetch_series_observations(series_id=series_id, **kwargs)

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate normalized FRED observations before storage."""
        missing = [
            column for column in NORMALIZED_COLUMNS if column not in data.columns
        ]
        if missing:
            msg = f"Missing required FRED observation columns: {missing}"
            raise ValueError(msg)

        validated = data.loc[:, NORMALIZED_COLUMNS].copy()
        if validated.empty:
            return validated

        if validated["series_id"].isna().any():
            msg = "series_id cannot contain missing values."
            raise ValueError(msg)
        if (validated["source"] != "fred").any():
            msg = "FRED observations must have source='fred'."
            raise ValueError(msg)

        validated["date"] = pd.to_datetime(validated["date"], errors="raise").dt.date
        validated["release_date"] = pd.to_datetime(
            validated["release_date"],
            errors="raise",
        ).dt.date
        validated["fetched_at"] = pd.to_datetime(
            validated["fetched_at"],
            errors="raise",
            utc=True,
        )
        validated["value"] = pd.to_numeric(validated["value"], errors="raise")

        if validated["date"].isna().any() or validated["release_date"].isna().any():
            msg = "date and release_date cannot contain missing values."
            raise ValueError(msg)
        return validated

    def store(
        self,
        data: pd.DataFrame,
        database_url: str | None = None,
    ) -> dict[str, int]:
        """Store validated FRED observations with idempotent upsert counts."""
        validated = self.validate(data)
        return upsert_observations(validated, database_url=database_url)


def _resolve_output_type(output_type: int | None, vintage_mode: str) -> int | None:
    if vintage_mode not in VALID_VINTAGE_MODES:
        msg = (
            f"Unsupported FRED vintage_mode: {vintage_mode}. "
            f"Supported values are {sorted(VALID_VINTAGE_MODES)}."
        )
        raise FredApiError(msg)
    if output_type is not None and output_type not in VALID_OUTPUT_TYPES:
        msg = (
            f"Unsupported FRED output_type: {output_type}. "
            f"Supported values are {sorted(VALID_OUTPUT_TYPES)}."
        )
        raise FredApiError(msg)
    if output_type is not None:
        return output_type
    if vintage_mode == "initial_release":
        return 4
    if vintage_mode == "realtime_period":
        return 1
    return None


def _extract_release_dates(
    frame: pd.DataFrame,
    payload: dict[str, Any],
    realtime_start: str | None,
    vintage_mode: str,
) -> pd.Series:
    if vintage_mode == "initial_release":
        if "realtime_start" not in frame.columns:
            msg = (
                "FRED initial_release mode requires per-observation "
                "realtime_start metadata."
            )
            raise FredApiError(msg)
        if frame["realtime_start"].isna().any():
            msg = (
                "FRED initial_release mode received observations with missing "
                "realtime_start metadata."
            )
            raise FredApiError(msg)
        return pd.to_datetime(frame["realtime_start"], errors="raise").dt.date

    release_source = frame.get("realtime_start")
    if release_source is None:
        release_source = realtime_start or payload.get("realtime_start")
    if release_source is None:
        msg = "FRED response did not include realtime_start metadata."
        raise FredApiError(msg)

    # In current-vintage mode, FRED may return the same realtime_start for every
    # historical observation. That is a snapshot/vintage proxy, not proof that
    # every historical value was available on that date. Use initial_release
    # mode for strict historical point-in-time feature construction.
    if isinstance(release_source, str):
        parsed = pd.to_datetime(release_source, errors="raise").date()
        return pd.Series([parsed] * len(frame), index=frame.index)
    return pd.to_datetime(release_source, errors="raise").dt.date
