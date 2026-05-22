"""FRED/ALFRED source client for point-in-time macro observations."""

import time
from collections.abc import Iterator
from datetime import UTC, date
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
FRED_INITIAL_RELEASE_REALTIME_START = "1776-07-04"
FRED_INITIAL_RELEASE_REALTIME_END = "9999-12-31"


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
        chunk_realtime: bool = False,
        realtime_chunk_years: int = 5,
    ) -> pd.DataFrame:
        """Fetch and normalize observations for one FRED/ALFRED series."""
        request_output_type = _resolve_output_type(output_type, vintage_mode)
        request_realtime_start, request_realtime_end = _resolve_realtime_window(
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            vintage_mode=vintage_mode,
        )
        fetched_at = pd.Timestamp.now(tz="UTC")
        if chunk_realtime:
            return self._fetch_chunked_initial_release(
                series_id=series_id,
                observation_start=observation_start,
                observation_end=observation_end,
                realtime_start=request_realtime_start,
                realtime_end=request_realtime_end,
                frequency=frequency,
                aggregation_method=aggregation_method,
                output_type=request_output_type,
                vintage_mode=vintage_mode,
                realtime_chunk_years=realtime_chunk_years,
                fetched_at=fetched_at,
                use_observation_start_floor=realtime_start is None,
            )

        payload = self._request_json(
            "series/observations",
            {
                "series_id": series_id,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "realtime_start": request_realtime_start,
                "realtime_end": request_realtime_end,
                "frequency": frequency,
                "aggregation_method": aggregation_method,
                "output_type": request_output_type,
            },
        )
        return _normalize_observations_payload(
            series_id=series_id,
            payload=payload,
            realtime_start=request_realtime_start,
            vintage_mode=vintage_mode,
            fetched_at=fetched_at,
        )

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

    def _fetch_chunked_initial_release(
        self,
        series_id: str,
        observation_start: str | None,
        observation_end: str | None,
        realtime_start: str | None,
        realtime_end: str | None,
        frequency: str | None,
        aggregation_method: str | None,
        output_type: int | None,
        vintage_mode: str,
        realtime_chunk_years: int,
        fetched_at: pd.Timestamp,
        use_observation_start_floor: bool,
    ) -> pd.DataFrame:
        if vintage_mode != "initial_release":
            msg = (
                "chunk_realtime is only supported with "
                "vintage_mode='initial_release'."
            )
            raise FredApiError(msg)
        if realtime_chunk_years < 1:
            msg = "realtime_chunk_years must be at least 1."
            raise FredApiError(msg)

        chunk_start, chunk_end = _resolve_chunk_bounds(
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            observation_start=observation_start,
            use_observation_start_floor=use_observation_start_floor,
        )
        frames: list[pd.DataFrame] = []
        for start, end in _iter_realtime_chunks(
            chunk_start,
            chunk_end,
            years=realtime_chunk_years,
        ):
            try:
                payload = self._request_json(
                    "series/observations",
                    {
                        "series_id": series_id,
                        "observation_start": observation_start,
                        "observation_end": observation_end,
                        "realtime_start": start,
                        "realtime_end": end,
                        "frequency": frequency,
                        "aggregation_method": aggregation_method,
                        "output_type": output_type,
                    },
                )
                normalized = _normalize_observations_payload(
                    series_id=series_id,
                    payload=payload,
                    realtime_start=start,
                    vintage_mode=vintage_mode,
                    fetched_at=fetched_at,
                )
            except FredApiError as exc:
                if _is_no_vintage_dates_error(exc):
                    continue
                msg = (
                    "FRED initial_release chunk failed for real-time window "
                    f"{start} to {end}: {exc}"
                )
                raise FredApiError(msg) from exc
            frames.append(normalized)

        if not frames:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)
        return (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )


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


def _resolve_realtime_window(
    realtime_start: str | None,
    realtime_end: str | None,
    vintage_mode: str,
) -> tuple[str | None, str | None]:
    if vintage_mode != "initial_release":
        return realtime_start, realtime_end

    # FRED otherwise defaults realtime_start/realtime_end to today's date.
    # For output_type=4 this can restrict initial-release observations to a
    # one-day current vintage window and omit historical releases.
    return (
        realtime_start or FRED_INITIAL_RELEASE_REALTIME_START,
        realtime_end or FRED_INITIAL_RELEASE_REALTIME_END,
    )


def _normalize_observations_payload(
    series_id: str,
    payload: dict[str, Any],
    realtime_start: str | None,
    vintage_mode: str,
    fetched_at: pd.Timestamp,
) -> pd.DataFrame:
    observations = payload.get("observations")
    if observations is None:
        msg = "FRED API response did not include an observations field."
        raise FredApiError(msg)
    if not observations:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    frame = pd.DataFrame(observations)
    values = frame["value"].replace({".": np.nan, "": np.nan})
    release_dates = _extract_release_dates(
        frame=frame,
        payload=payload,
        realtime_start=realtime_start,
        vintage_mode=vintage_mode,
    )
    return pd.DataFrame(
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


def _resolve_chunk_bounds(
    realtime_start: str | None,
    realtime_end: str | None,
    observation_start: str | None,
    use_observation_start_floor: bool,
) -> tuple[date, date]:
    if realtime_start is None or realtime_end is None:
        msg = "Chunked FRED requests require an effective real-time window."
        raise FredApiError(msg)

    start = pd.Timestamp(realtime_start).date()
    end = pd.Timestamp(realtime_end).date()
    today = pd.Timestamp.now(tz=UTC).date()
    if end > today:
        end = today

    if use_observation_start_floor and observation_start is not None:
        observation_floor = pd.Timestamp(observation_start).date()
        if observation_floor > start:
            start = observation_floor

    if start > end:
        msg = f"Invalid FRED chunk real-time window: {start} to {end}."
        raise FredApiError(msg)
    return start, end


def _iter_realtime_chunks(
    start: date,
    end: date,
    years: int,
) -> Iterator[tuple[str, str]]:
    current = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while current <= final:
        next_start = current + pd.DateOffset(years=years)
        chunk_end = min(next_start - pd.Timedelta(days=1), final)
        yield current.date().isoformat(), chunk_end.date().isoformat()
        current = chunk_end + pd.Timedelta(days=1)


def _is_no_vintage_dates_error(exc: FredApiError) -> bool:
    return "no vintage dates exist" in str(exc).lower()


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
