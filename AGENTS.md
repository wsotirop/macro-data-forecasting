# Agent Instructions

- Do not hardcode absolute paths.
- Do not commit API keys, credentials, or `.env` files.
- Do not implement lookahead-biased features.
- Every public function needs type hints and a one-line docstring.
- Prefer small, testable modules.
- Use `release_date` rather than `reference_date` for point-in-time availability.
- Use walk-forward validation only for time-series modeling.
- Do not use k-fold cross-validation for macro forecasting.
- If a data availability assumption is unclear, document it clearly.
- Do not silently swallow errors.
