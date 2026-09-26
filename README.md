# Eid holidays and PM2.5 in Dhaka — code and data

Supplementary archive for: "Does Dhaka's Air Get Cleaner When the City Empties? A Nine-Year
Natural Experiment Using Eid Holidays and Reference-Grade PM2.5 Data" (H. Debnath),
submitted to the Columbia Junior Science Journal, 2026-27 cycle.

## Contents

- `eid_analysis.py` — cleaning, daily aggregation, event study, pooled estimate (bootstrap CI,
  t-test, Wilcoxon), placebo/permutation test, sensitivity analyses, hourly profile, weekend
  benchmark. Writes everything in `outputs/`.
- `make_figures.py` — Figures 1, 2 and S1 from the files in `outputs/`.
- `make_figures_col.py` — column-width versions of Figures 1 and 2 used in the manuscript.
- `weather_control.py` — merges NOAA GSOD Dhaka station weather (2016–2020) with the daily
  PM2.5 series and re-estimates the Eid effect three ways: a weather balance test, a day-level
  regression adjusting for temperature/wind/precipitation/humidity, and a random-forest
  "deweathering" model. Writes `outputs/weather_control.json` and `outputs/weather_merged.csv`.
- `make_figS2.py` — Figure S2 (estimates with and without weather adjustment).
- `DebnathHrishin_supplement.docx` — supplementary material: Tables S1 (Eid date sources),
  S2 (per-holiday results), S3 (weather control), Figures S1 and S2, and methods notes.
- `raw_data/` — the ten AirNow "Embassy Historical" hourly PM2.5 files for Dhaka
  (Dhaka_PM2.5_YYYY_YTD.csv, 2016–2025), copied unchanged from
  https://github.com/dolekhanhdang/Air-Quality-Data-from-U.S.-Embassies (EPA-provided archive).
- `raw_data/gsod_dhaka/` — five GSOD `.op` weather files for Dhaka (station 419230), 2016–2020,
  from the NOAA archive mirror https://github.com/CalebBell/gsod
- `outputs/` — processed daily series, per-holiday results, event-study curve, hourly profiles,
  `results.json` (every number quoted in the paper), and the analysis log.
- `figures/` — PNG (300 dpi) and PDF versions of the figures.

## Reproduce

Python 3.10+ with pandas, numpy, scipy, matplotlib (and scikit-learn, py7zr for the weather step).
Edit the path constants at the top of each script, then run:

```
python eid_analysis.py
python make_figures.py
python weather_control.py
python make_figS2.py
```

Random seed 20260922 is fixed; the main run takes about one minute.

## AI use

During the preparation of this work, Anthropic’s Claude (Opus 4.8) was used to assist with ideation and initial manuscript drafting.
