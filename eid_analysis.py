
import glob, json, os
import numpy as np
import pandas as pd
from scipy import stats

DATA_GLOB = "/home/embassy_aq/Data/Dhaka/*/Dhaka_PM2.5_*_YTD.csv"
OUT = "/home/work"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260922)

# ---------------------------------------------------------------- Eid dates (Bangladesh)
# Verified against Bangladesh National Moon Sighting Committee announcements reported by
# Dhaka Tribune / Prothom Alo / UNB / Financial Express (Fitr) and timeanddate.com (Adha).
EIDS = [
    ("2016-07-07", "Fitr"), ("2016-09-13", "Adha"),
    ("2017-06-26", "Fitr"), ("2017-09-02", "Adha"),
    ("2018-06-16", "Fitr"), ("2018-08-22", "Adha"),
    ("2019-06-05", "Fitr"), ("2019-08-12", "Adha"),
    ("2020-05-25", "Fitr"), ("2020-08-01", "Adha"),
    ("2021-05-14", "Fitr"), ("2021-07-21", "Adha"),
    ("2022-05-03", "Fitr"), ("2022-07-10", "Adha"),
    ("2023-04-22", "Fitr"), ("2023-06-29", "Adha"),
    ("2024-04-11", "Fitr"), ("2024-06-17", "Adha"),
]
EIDS = [(pd.Timestamp(d), k) for d, k in EIDS]

# Window definitions (days relative to Eid day, inclusive)
EID_WIN = (-1, 3)            # primary: Eid eve + Eid + 3 days after (official 3-day holiday + 2)
REF_WIN = [(-21, -8), (8, 21)]  # reference: two weeks either side, 7-day buffer skipped
MIN_HOURS = 18               # min valid hours for a daily mean
MIN_EID_DAYS = 4             # min valid days inside the Eid window
MIN_REF_DAYS = 14            # min valid days inside the reference windows

# ---------------------------------------------------------------- load & clean
files = sorted(glob.glob(DATA_GLOB))
raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
raw["dt"] = pd.to_datetime(raw["Date (LT)"], format="%Y-%m-%d %I:%M %p", errors="coerce")
raw = raw.dropna(subset=["dt"]).drop_duplicates("dt").sort_values("dt")
ok = (raw["QC Name"] == "Valid") & (raw["Raw Conc."] > 0)
raw["pm25"] = np.where(ok, raw["Raw Conc."], np.nan)
hourly = raw.set_index("dt")["pm25"]
hourly = hourly.reindex(pd.date_range(hourly.index.min().floor("D"), hourly.index.max().ceil("D"), freq="h"))

# The AirNow convention: the timestamp is the END of the hourly averaging period (01:00 = 00:00-01:00).
# Shift back 1 hour so that a calendar day holds its own 24 hours.
hourly.index = hourly.index - pd.Timedelta(hours=1)
hourly = hourly.rename_axis("dt")

daily_mean = hourly.resample("D").mean()
daily_n = hourly.resample("D").count()
daily = pd.DataFrame({"pm25": daily_mean, "n_hours": daily_n})
daily.loc[daily["n_hours"] < MIN_HOURS, "pm25"] = np.nan
daily["date"] = daily.index

summary = {
    "hourly_rows_raw": int(len(raw)),
    "hourly_valid": int(np.isfinite(raw["pm25"]).sum()),
    "first_obs": str(raw["dt"].min()), "last_obs": str(raw["dt"].max()),
    "daily_days_total": int(len(daily)), "daily_days_valid": int(daily["pm25"].notna().sum()),
    "hourly_mean": float(np.nanmean(raw["pm25"])), "hourly_median": float(np.nanmedian(raw["pm25"])),
}

# ---------------------------------------------------------------- event study
def window_mean(series_by_t, lo, hi):
    vals = [series_by_t.get(t, np.nan) for t in range(lo, hi + 1)]
    vals = np.array(vals, dtype=float)
    return np.nanmean(vals) if np.isfinite(vals).sum() > 0 else np.nan, int(np.isfinite(vals).sum())

def event_stats(eid_date, eid_win=EID_WIN, ref_win=REF_WIN):
    d = daily["pm25"]
    s = {}
    for t in range(-30, 31):
        day = eid_date + pd.Timedelta(days=t)
        s[t] = d.get(day, np.nan) if day in d.index else np.nan
    eid_mean, eid_n = window_mean(s, *eid_win)
    ref_vals, ref_n = [], 0
    for lo, hi in ref_win:
        m, n = window_mean(s, lo, hi)
        ref_vals += [s.get(t, np.nan) for t in range(lo, hi + 1)]
        ref_n += n
    ref_vals = np.array(ref_vals, dtype=float)
    ref_mean = np.nanmean(ref_vals) if ref_n > 0 else np.nan
    pre_mean, pre_n = window_mean(s, ref_win[0][0], ref_win[0][1])
    if len(ref_win) > 1:
        post_mean, post_n = window_mean(s, ref_win[1][0], ref_win[1][1])
    else:
        post_mean, post_n = np.nan, 0
    return dict(eid_mean=eid_mean, eid_n=eid_n, ref_mean=ref_mean, ref_n=ref_n,
                pre_mean=pre_mean, pre_n=pre_n, post_mean=post_mean, post_n=post_n, series=s)

rows, series_rel = [], {}
for eid_date, kind in EIDS:
    es = event_stats(eid_date)
    valid = (es["eid_n"] >= MIN_EID_DAYS) and (es["ref_n"] >= MIN_REF_DAYS)
    lr = np.log(es["eid_mean"] / es["ref_mean"]) if valid else np.nan
    rows.append(dict(
        eid=eid_date.date().isoformat(), kind=kind, year=eid_date.year, month=eid_date.month,
        weekday=eid_date.day_name(), eid_mean=es["eid_mean"], eid_days=es["eid_n"],
        ref_mean=es["ref_mean"], ref_days=es["ref_n"], pre_mean=es["pre_mean"], post_mean=es["post_mean"],
        abs_change=es["eid_mean"] - es["ref_mean"] if valid else np.nan,
        pct_change=100 * (es["eid_mean"] / es["ref_mean"] - 1) if valid else np.nan,
        log_ratio=lr, included=valid,
    ))
    if valid:
        # daily series as % of the event's own reference mean
        series_rel[eid_date.date().isoformat()] = {t: 100 * v / es["ref_mean"] for t, v in es["series"].items()}

ev = pd.DataFrame(rows)
inc = ev[ev["included"]].copy()

def pooled(lrs, n_boot=10000):
    lrs = np.asarray(lrs, dtype=float)
    lrs = lrs[np.isfinite(lrs)]
    n = len(lrs)
    mean_lr = lrs.mean()
    boots = np.array([RNG.choice(lrs, n, replace=True).mean() for _ in range(n_boot)])
    ci = np.percentile(boots, [2.5, 97.5])
    t_p = stats.ttest_1samp(lrs, 0.0).pvalue
    w_p = stats.wilcoxon(lrs, alternative="two-sided").pvalue if n >= 6 else np.nan
    return dict(n=int(n), mean_log_ratio=float(mean_lr), pct=float(100 * (np.exp(mean_lr) - 1)),
                ci_pct=[float(100 * (np.exp(ci[0]) - 1)), float(100 * (np.exp(ci[1]) - 1))],
                t_test_p=float(t_p), wilcoxon_p=float(w_p), n_negative=int((lrs < 0).sum()),
                median_pct=float(100 * (np.exp(np.median(lrs)) - 1)))

results = {"data": summary, "windows": {"eid": EID_WIN, "ref": REF_WIN, "min_hours": MIN_HOURS}}
results["pooled_all"] = pooled(inc["log_ratio"])
results["pooled_fitr"] = pooled(inc.loc[inc["kind"] == "Fitr", "log_ratio"])
results["pooled_adha"] = pooled(inc.loc[inc["kind"] == "Adha", "log_ratio"])
results["pooled_excl_covid"] = pooled(inc.loc[~inc["year"].isin([2020, 2021]), "log_ratio"])
results["pooled_covid_only"] = pooled(inc.loc[inc["year"].isin([2020, 2021]), "log_ratio"])
results["abs_change_mean"] = float(inc["abs_change"].mean())
results["abs_change_median"] = float(inc["abs_change"].median())
results["eid_mean_avg"] = float(inc["eid_mean"].mean())
results["ref_mean_avg"] = float(inc["ref_mean"].mean())
# Fitr vs Adha difference test
results["fitr_vs_adha_p"] = float(stats.mannwhitneyu(inc.loc[inc.kind == "Fitr", "log_ratio"],
                                                     inc.loc[inc.kind == "Adha", "log_ratio"]).pvalue)

# ---------------------------------------------------------------- sensitivity to window choice
sens = {}
for name, ew, rw in [
    ("primary_-1_+3", (-1, 3), REF_WIN),
    ("holiday_only_-1_+1", (-1, 1), REF_WIN),
    ("wide_-2_+5", (-2, 5), REF_WIN),
    ("eid_day_only_0_0", (0, 0), REF_WIN),
    ("short_ref_-14_-8_+8_+14", (-1, 3), [(-14, -8), (8, 14)]),
    ("pre_only_ref_-21_-8", (-1, 3), [(-21, -8)]),
]:
    lrs = []
    for eid_date, kind in EIDS:
        es = event_stats(eid_date, ew, rw)
        need_ref = MIN_REF_DAYS if len(rw) == 2 else 7
        need_eid = max(1, int(0.7 * (ew[1] - ew[0] + 1)))
        if es["eid_n"] >= need_eid and es["ref_n"] >= need_ref:
            lrs.append(np.log(es["eid_mean"] / es["ref_mean"]))
    sens[name] = pooled(lrs, n_boot=10000)
results["sensitivity"] = sens

# ---------------------------------------------------------------- pre-Eid rush (t = -7..-2)
lrs = []
for eid_date, kind in EIDS:
    es = event_stats(eid_date, (-7, -2), REF_WIN)
    if es["eid_n"] >= 4 and es["ref_n"] >= MIN_REF_DAYS:
        lrs.append(np.log(es["eid_mean"] / es["ref_mean"]))
results["pre_eid_rush_-7_-2"] = pooled(lrs, n_boot=10000)
# return period t = +4..+7
lrs = []
for eid_date, kind in EIDS:
    es = event_stats(eid_date, (4, 7), REF_WIN)
    if es["eid_n"] >= 3 and es["ref_n"] >= MIN_REF_DAYS:
        lrs.append(np.log(es["eid_mean"] / es["ref_mean"]))
results["return_period_+4_+7"] = pooled(lrs, n_boot=10000)

# ---------------------------------------------------------------- placebo / permutation test
eid_dates_all = [d for d, _ in EIDS]
def is_far_from_eid(day, buffer=28):
    return all(abs((day - e).days) > buffer for e in eid_dates_all)

years = sorted(set(d.year for d, _ in EIDS))
candidate_days = {}
for y in years:
    days = pd.date_range(f"{y}-01-22", f"{y}-12-10", freq="D")  # keep +-21 day windows inside the record
    cand = [d for d in days if is_far_from_eid(d) and (d + pd.Timedelta(days=21)) <= daily.index.max()]
    candidate_days[y] = cand

n_perm = 2000
placebo_means = []
n_events = len(inc)
event_years = list(inc["year"])
for _ in range(n_perm):
    lrs = []
    for y in event_years:
        for _try in range(20):
            pseudo = candidate_days[y][RNG.integers(len(candidate_days[y]))]
            es = event_stats(pseudo)
            if es["eid_n"] >= MIN_EID_DAYS and es["ref_n"] >= MIN_REF_DAYS:
                lrs.append(np.log(es["eid_mean"] / es["ref_mean"]))
                break
    if len(lrs) >= n_events - 2:
        placebo_means.append(np.mean(lrs))
placebo_means = np.array(placebo_means)
obs = results["pooled_all"]["mean_log_ratio"]
results["placebo"] = dict(
    n_perm=int(len(placebo_means)), placebo_mean_pct=float(100 * (np.exp(placebo_means.mean()) - 1)),
    placebo_sd_log=float(placebo_means.std()),
    p_one_sided=float((placebo_means <= obs).mean()),
    placebo_pct_2_5=float(100 * (np.exp(np.percentile(placebo_means, 2.5)) - 1)),
    placebo_pct_97_5=float(100 * (np.exp(np.percentile(placebo_means, 97.5)) - 1)),
    min_placebo_pct=float(100 * (np.exp(placebo_means.min()) - 1)),
)
np.save(f"{OUT}/placebo_means.npy", placebo_means)

# ---------------------------------------------------------------- event-study curve (t = -21..+21)
curve_rows = []
for t in range(-21, 22):
    vals = np.array([series_rel[k].get(t, np.nan) for k in series_rel], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) >= 5:
        boots = np.array([RNG.choice(vals, len(vals), replace=True).mean() for _ in range(3000)])
        curve_rows.append(dict(t=t, mean_pct=vals.mean(), lo=np.percentile(boots, 2.5),
                               hi=np.percentile(boots, 97.5), n=len(vals), median_pct=np.median(vals)))
curve = pd.DataFrame(curve_rows)
curve.to_csv(f"{OUT}/event_study_curve.csv", index=False)
results["curve_min"] = dict(t=int(curve.loc[curve["mean_pct"].idxmin(), "t"]),
                            mean_pct=float(curve["mean_pct"].min()))
results["curve_by_t"] = {int(r.t): round(float(r.mean_pct), 1) for r in curve.itertuples()}

# recovery: first t>0 where 3-day centred mean of curve >= 95% of baseline
c = curve.set_index("t")["mean_pct"]
roll = c.rolling(3, center=True).mean()
rec = [t for t in range(1, 22) if t in roll.index and roll[t] >= 95]
results["recovery_day_95pct"] = int(rec[0]) if rec else None

# ---------------------------------------------------------------- diurnal profile
hourly_df = hourly.to_frame("pm25")
hourly_df["date"] = hourly_df.index.floor("D")
hourly_df["hour"] = hourly_df.index.hour
eid_days, ref_days = set(), set()
for eid_date, kind in EIDS:
    if not ev.loc[ev["eid"] == eid_date.date().isoformat(), "included"].iloc[0]:
        continue
    for t in range(EID_WIN[0], EID_WIN[1] + 1):
        eid_days.add(eid_date + pd.Timedelta(days=t))
    for lo, hi in REF_WIN:
        for t in range(lo, hi + 1):
            ref_days.add(eid_date + pd.Timedelta(days=t))
# Express every hour as % of that event's reference mean so events with different seasons pool cleanly
ref_mean_by_event = {pd.Timestamp(r.eid): r.ref_mean for r in inc.itertuples()}
def event_of(day):
    for e in ref_mean_by_event:
        if abs((day - e).days) <= 21:
            return e
    return None
hourly_df["event"] = hourly_df["date"].map(lambda d: event_of(d) if (d in eid_days or d in ref_days) else None)
sub = hourly_df.dropna(subset=["event"]).copy()
sub["rel"] = 100 * sub["pm25"] / sub["event"].map(ref_mean_by_event)
sub["group"] = np.where(sub["date"].isin(eid_days), "eid", "ref")
diurnal = sub.groupby(["group", "hour"])["rel"].mean().unstack(0)
diurnal_abs = sub.groupby(["group", "hour"])["pm25"].mean().unstack(0)
diurnal.to_csv(f"{OUT}/diurnal_relative.csv")
diurnal_abs.to_csv(f"{OUT}/diurnal_absolute.csv")
diurnal["reduction_pct"] = 100 * (1 - diurnal["eid"] / diurnal["ref"])
results["diurnal"] = {
    "reduction_by_hour_pct": {int(h): round(float(v), 1) for h, v in diurnal["reduction_pct"].items()},
    "max_reduction_hour": int(diurnal["reduction_pct"].idxmax()),
    "max_reduction_pct": float(diurnal["reduction_pct"].max()),
    "min_reduction_hour": int(diurnal["reduction_pct"].idxmin()),
    "min_reduction_pct": float(diurnal["reduction_pct"].min()),
    "evening_20_23_reduction_pct": float(diurnal.loc[20:23, "reduction_pct"].mean()),
    "morning_7_10_reduction_pct": float(diurnal.loc[7:10, "reduction_pct"].mean()),
    "night_0_5_reduction_pct": float(diurnal.loc[0:5, "reduction_pct"].mean()),
    "afternoon_12_16_reduction_pct": float(diurnal.loc[12:16, "reduction_pct"].mean()),
}

# ---------------------------------------------------------------- benchmark: ordinary weekend effect
daily["dow"] = daily.index.dayofweek
daily["month"] = daily.index.month
# Compare Friday (BD weekend) with Sun-Thu within the same month-year, non-Eid periods
non_eid = daily[~daily.index.isin(list(eid_days))].copy()
non_eid["ym"] = non_eid.index.to_period("M")
g = non_eid.groupby(["ym", non_eid["dow"].isin([4, 5])])["pm25"].mean().unstack()
g = g.dropna()
weekend_lr = np.log(g[True] / g[False])
results["weekend_effect"] = dict(n_months=int(len(weekend_lr)), pct=float(100 * (np.exp(weekend_lr.mean()) - 1)),
                                 ci_pct=[float(100 * (np.exp(x) - 1)) for x in
                                         np.percentile([RNG.choice(weekend_lr, len(weekend_lr)).mean() for _ in range(4000)], [2.5, 97.5])])

# ---------------------------------------------------------------- WHO / AQI context on Eid days
who_24h = 15.0
eid_daily = daily.loc[daily.index.isin(list(eid_days)), "pm25"].dropna()
ref_daily = daily.loc[daily.index.isin(list(ref_days)), "pm25"].dropna()
results["who_context"] = dict(
    eid_days_n=int(len(eid_daily)), eid_days_above_who24=float(100 * (eid_daily > who_24h).mean()),
    ref_days_n=int(len(ref_daily)), ref_days_above_who24=float(100 * (ref_daily > who_24h).mean()),
    eid_days_above_us_35=float(100 * (eid_daily > 35.4).mean()), ref_days_above_us_35=float(100 * (ref_daily > 35.4).mean()),
    eid_daily_mean=float(eid_daily.mean()), ref_daily_mean=float(ref_daily.mean()),
)

ev.to_csv(f"{OUT}/per_event_results.csv", index=False)
daily.to_csv(f"{OUT}/daily_pm25_dhaka_embassy.csv")
with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(json.dumps({k: v for k, v in results.items() if k not in ("curve_by_t",)}, indent=1, default=float))
print(ev.to_string())
