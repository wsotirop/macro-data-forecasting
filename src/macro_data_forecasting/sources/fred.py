"""FRED/ALFRED source client for point-in-time macro observations."""

import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from macro_data_forecasting.config import get_settings
from macro_data_forecasting.database import (
    REQUIRED_OBSERVATION_COLUMNS,
    insert_observations,
)

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
NORMALIZED_COLUMNS = list(REQUIRED_OBSERVATION_COLUMNS)


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
    ) -> pd.DataFrame:
        """Fetch and normalize observations for one FRED/ALFRED series."""
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

        # FRED observations include ALFRED realtime/vintage metadata, but the
        # observations endpoint does not provide an exact release calendar date
        # for every economic reference period. Until explicit release calendars
        # are integrated, realtime_start is the best available availability date.
        release_source = frame.get("realtime_start")
        if release_source is None:
            release_source = realtime_start or payload.get("realtime_start")
        if release_source is None:
            msg = "FRED response did not include realtime_start metadata."
            raise FredApiError(msg)

        normalized = pd.DataFrame(
            {
                "series_id": series_id,
                "date": pd.to_datetime(frame["date"], errors="raise").dt.date,
                "value": pd.to_numeric(values, errors="raise"),
                "source": "fred",
                "release_date": pd.to_datetime(
                    release_source,
                    errors="raise",
                ).date
                if isinstance(release_source, str)
                else pd.to_datetime(release_source, errors="raise").dt.date,
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

    def store(self, data: pd.DataFrame, database_url: str | None = None) -> int:
        """Store validated FRED observations and return the inserted row count."""
        validated = self.validate(data)
        return insert_observations(validated, database_url=database_url)
