"""BLS source client for CPI-related macro observations."""

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

BLS_API_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
NORMALIZED_COLUMNS = list(REQUIRED_OBSERVATION_COLUMNS)


class BlsApiError(RuntimeError):
    """Raised when the BLS API request or response is invalid."""


class BlsClient:
    """Client for fetching normalized BLS observations."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BLS_API_BASE_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ) -> None:
        """Initialize a BLS client with optional settings-backed credentials."""
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.bls_api_key
        self.base_url = base_url
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = {
            key: value for key, value in payload.items() if value is not None
        }
        if self.api_key:
            request_payload["registrationkey"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.base_url,
                    json=request_payload,
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
                    "BLS API request failed with status "
                    f"{response.status_code}: {response.text}"
                )
                raise BlsApiError(msg)

            try:
                response_payload = response.json()
            except ValueError as exc:
                msg = "BLS API response was not valid JSON."
                raise BlsApiError(msg) from exc

            status = response_payload.get("status")
            if status != "REQUEST_SUCCEEDED":
                message = response_payload.get("message")
                msg = f"BLS API returned status {status!r}: {message}"
                raise BlsApiError(msg)
            return response_payload

        if last_error is not None:
            msg = "BLS API request failed after retries."
            raise BlsApiError(msg) from last_error
        msg = "BLS API request failed after retries due to transient HTTP errors."
        raise BlsApiError(msg)

    @staticmethod
    def _monthly_dates(frame: pd.DataFrame) -> pd.Series:
        period = frame["period"].astype(str)
        valid_period = period.str.fullmatch(r"M(0[1-9]|1[0-2])")
        if not valid_period.all():
            bad_periods = sorted(period.loc[~valid_period].unique())
            msg = (
                "BLS response contained non-monthly periods that cannot be "
                f"stored as monthly observations: {bad_periods}"
            )
            raise BlsApiError(msg)

        months = period.str.removeprefix("M")
        date_strings = frame["year"].astype(str) + "-" + months + "-01"
        return pd.to_datetime(date_strings, errors="raise").dt.date

    def fetch_series_observations(
        self,
        series_id: str,
        start_year: int,
        end_year: int,
        annual_average: bool = False,
    ) -> pd.DataFrame:
        """Fetch and normalize monthly observations for one BLS series."""
        payload = self._request_json(
            {
                "seriesid": [series_id],
                "startyear": str(start_year),
                "endyear": str(end_year),
                "annualaverage": "true" if annual_average else "false",
            },
        )
        series = payload.get("Results", {}).get("series")
        if not series:
            msg = "BLS API response did not include Results.series data."
            raise BlsApiError(msg)

        series_payload = next(
            (
                item
                for item in series
                if item.get("seriesID") == series_id
                or item.get("seriesId") == series_id
            ),
            None,
        )
        if series_payload is None:
            msg = f"BLS API response did not include requested series {series_id}."
            raise BlsApiError(msg)

        observations = series_payload.get("data")
        if observations is None:
            msg = f"BLS API response for {series_id} did not include data."
            raise BlsApiError(msg)
        if not observations:
            return pd.DataFrame(columns=NORMALIZED_COLUMNS)

        frame = pd.DataFrame(observations)
        required_api_columns = {"year", "period", "value"}
        missing = required_api_columns.difference(frame.columns)
        if missing:
            msg = f"BLS API response missing observation fields: {sorted(missing)}"
            raise BlsApiError(msg)

        values = frame["value"].replace({".": np.nan, "": np.nan})
        normalized = pd.DataFrame(
            {
                "series_id": series_id,
                "date": self._monthly_dates(frame),
                "value": pd.to_numeric(values, errors="coerce"),
                "source": "bls",
                "release_date": pd.NaT,
                "fetched_at": pd.Timestamp.now(tz="UTC"),
            },
            columns=NORMALIZED_COLUMNS,
        )
        return normalized

    def fetch(self, series_id: str, **kwargs: Any) -> pd.DataFrame:
        """Fetch normalized BLS observations for a source-client workflow."""
        return self.fetch_series_observations(series_id=series_id, **kwargs)

    def validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate normalized BLS observations before storage."""
        missing = [
            column for column in NORMALIZED_COLUMNS if column not in data.columns
        ]
        if missing:
            msg = f"Missing required BLS observation columns: {missing}"
            raise ValueError(msg)

        validated = data.loc[:, NORMALIZED_COLUMNS].copy()
        if validated.empty:
            return validated

        if validated["series_id"].isna().any():
            msg = "series_id cannot contain missing values."
            raise ValueError(msg)
        if (validated["source"] != "bls").any():
            msg = "BLS observations must have source='bls'."
            raise ValueError(msg)

        validated["date"] = pd.to_datetime(validated["date"], errors="raise").dt.date
        release_dates = pd.to_datetime(validated["release_date"], errors="coerce")
        invalid_release_date = validated["release_date"].notna() & release_dates.isna()
        if invalid_release_date.any():
            msg = "release_date contains invalid date values."
            raise ValueError(msg)
        validated["release_date"] = release_dates.dt.date
        validated["fetched_at"] = pd.to_datetime(
            validated["fetched_at"],
            errors="raise",
            utc=True,
        )
        validated["value"] = pd.to_numeric(validated["value"], errors="coerce")

        if validated["date"].isna().any():
            msg = "date cannot contain missing values."
            raise ValueError(msg)
        return validated

    def store(
        self,
        data: pd.DataFrame,
        database_url: str | None = None,
    ) -> dict[str, int]:
        """Store validated BLS observations with idempotent upsert counts."""
        validated = self.validate(data)
        return upsert_observations(validated, database_url=database_url)
