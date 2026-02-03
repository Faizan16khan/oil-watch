# oil-watch
A time-series monitoring project for crude oil markets, designed to study regime shifts, noise characteristics, and signal decay under non-stationary conditions. Emphasizes robust evaluation, causal plausibility, and extensibility toward satellite imagery and multi-modal signals.
# Oil Watch

Satellite + data pipeline project to track oil storage signals and turn them into ML-ready features.

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Fetch scenes (test)
python -m src.stac_fetch