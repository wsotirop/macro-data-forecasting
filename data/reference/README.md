# Reference Data

This directory is for small, versioned reference files that define data availability rules.

CPI release dates matter because CPI observations are not usable for strict point-in-time modeling until the official release date is known. The economic reference month and the public release date are different fields, and using the reference month as the availability date would create lookahead bias.

Expected CPI release calendar schema:

```text
release_name,reference_period,release_date,release_time
Consumer Price Index,2026-04,2026-05-12,08:30
```

- `release_name`: release label, such as `Consumer Price Index`.
- `reference_period`: CPI reference month in `YYYY-MM` format.
- `release_date`: official release date in `YYYY-MM-DD` format.
- `release_time`: release time as a string, such as `08:30`.

`cpi_release_calendar_sample.csv` is only a sample for tests and examples. It is not a complete historical CPI release calendar.

Before real point-in-time CPI backtests, provide a complete official release calendar covering the full observation window. Do not silently approximate missing release dates.
