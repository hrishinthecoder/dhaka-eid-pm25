# Eid holidays and PM2.5 in Dhaka — code and data

Supplementary archive for: "Does Dhaka's Air Get Cleaner When the City Empties? A Nine-Year
Natural Experiment Using Eid Holidays and Reference-Grade PM2.5 Data" (H. Debnath, F. Ahmed),
submitted to the Columbia Junior Science Journal, 2026-27 cycle.

## Contents
- `eid_analysis.py` — cleaning, daily aggregation, event study, pooled estimate (bootstrap CI,
  t-test, Wilcoxon), placebo/permutation test, sensitivity analyses, hourly profile, weekend
  benchmark. Writes everything in `outputs/`.
- `make_figures.py` — Figures 1, 2 and S1 from the files in `outputs/`.
- `raw_data/` — the ten AirNow "Embassy Historical" hourly PM2.5 files for Dhaka
  (Dhaka_PM2.5_<year>_YTD.csv, 2016–2025), copied unchanged from
  https://github.com/dolekhanhdang/Air-Quality-Data-from-U.S.-Embassies (EPA-provided archive).
- `outputs/` — processed daily series, per-holiday results, event-study curve, hourly profiles,
  `results.json` (every number quoted in the paper), and the analysis log.
- `figures/` — PNG (300 dpi) and PDF versions of the figures.

## Reproduce
Python 3.10+ with pandas, numpy, scipy, matplotlib.
Edit the two path constants at the top of each script (`DATA_GLOB`, `OUT`, `W`) to match your
folders, then run:

    python eid_analysis.py
    python make_figures.py

Random seed 20260922 is fixed; the run takes about one minute.

## Eid dates
Hard-coded in `eid_analysis.py` (`EIDS`). Sources for each date are listed in Supplementary Table S1.

## Weather control (added)
- `weather_control.py` — merges NOAA GSOD Dhaka station weather (2016–2020) with the daily
  PM2.5 series and re-estimates the Eid effect three ways: a weather balance test, a day-level
  regression adjusting for temperature/wind/precipitation/humidity, and a random-forest
  "deweathering" model. Writes `outputs/weather_control.json` and `outputs/weather_merged.csv`.
- `make_figS2.py` — Figure S2 (estimates with/without weather adjustment).
- `raw_data/gsod_dhaka/` — the five GSOD `.op` files for Dhaka (station 419230), 2016–2020,
  from the NOAA archive mirror https://github.com/CalebBell/gsod
- Needs `scikit-learn` and `py7zr` in addition to the packages above:
  `pip install scikit-learn py7zr`
- The full-period version should use the parent project's ERA5 reanalysis fields (all years),
  which were not available in this environment.
