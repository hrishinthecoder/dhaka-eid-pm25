
import glob, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

W = "/home/work"
RNG = np.random.default_rng(20260922)

# ---------------------------------------------------------------- GSOD weather
# GSOD .op fixed-width fields (documented positions), values in imperial units.
colspecs = [(0,6),(14,22),(24,30),(35,41),(78,83),(88,93),(118,123),(123,124)]
names = ["stn","yearmoda","temp_f","dewp_f","wdsp_kn","mxspd_kn","prcp_in","prcp_flag"]
rows = []
for f in sorted(glob.glob("/home/gsod_dhaka/*/419230-99999-*.op")):
    d = pd.read_fwf(f, colspecs=colspecs, names=names, skiprows=1)
    rows.append(d)
g = pd.concat(rows, ignore_index=True)
g["date"] = pd.to_datetime(g["yearmoda"].astype(str), format="%Y%m%d")
# GSOD missing codes
g["temp_f"] = g["temp_f"].where(g["temp_f"] < 9999, np.nan)
g["dewp_f"] = g["dewp_f"].where(g["dewp_f"] < 9999, np.nan)
g["wdsp_kn"] = g["wdsp_kn"].where(g["wdsp_kn"] < 999, np.nan)
g["mxspd_kn"] = g["mxspd_kn"].where(g["mxspd_kn"] < 999, np.nan)
g["prcp_in"] = g["prcp_in"].where(g["prcp_in"] < 99, np.nan)
# PRCP flag 'I' means no measurable precip reported -> treat as 0; else keep
g.loc[g["prcp_flag"].astype(str).str.strip() == "I", "prcp_in"] = 0.0
# convert to SI
g["temp_c"] = (g["temp_f"] - 32) * 5 / 9
g["dewp_c"] = (g["dewp_f"] - 32) * 5 / 9
g["wind_ms"] = g["wdsp_kn"] * 0.514444
g["maxwind_ms"] = g["mxspd_kn"] * 0.514444
g["precip_mm"] = g["prcp_in"] * 25.4
# relative humidity from temp and dew point (Magnus)
def rh(t, td):
    a, b = 17.625, 243.04
    return 100 * np.exp((a * td) / (b + td)) / np.exp((a * t) / (b + t))
g["rh_pct"] = rh(g["temp_c"], g["dewp_c"])
wx = g[["date","temp_c","dewp_c","wind_ms","maxwind_ms","precip_mm","rh_pct"]].set_index("date").sort_index()
wx_cov = ["temp_c","wind_ms","maxwind_ms","precip_mm","rh_pct"]

# ---------------------------------------------------------------- PM2.5 daily
pm = pd.read_csv(f"{W}/daily_pm25_dhaka_embassy.csv", index_col=0, parse_dates=True)
df = pm[["pm25"]].join(wx, how="left")
df = df.loc["2016-01-01":"2020-12-31"].copy()
df["log_pm25"] = np.log(df["pm25"])
df["precip_log"] = np.log1p(df["precip_mm"])
wx_cov2 = ["temp_c","wind_ms","maxwind_ms","precip_log","rh_pct"]
wx_coverage = df[wx_cov2].notna().all(axis=1).mean()

# ---------------------------------------------------------------- Eid windows
EIDS = [
    ("2016-07-07","Fitr"),("2016-09-13","Adha"),("2017-06-26","Fitr"),("2017-09-02","Adha"),
    ("2018-06-16","Fitr"),("2019-06-05","Fitr"),("2019-08-12","Adha"),("2020-08-01","Adha"),
]  # the 8 holidays that are usable AND fall in 2016-2020
EIDS = [(pd.Timestamp(d), k) for d, k in EIDS]
EID_WIN, REF = (-1, 3), [(-21, -8), (8, 21)]

def flags(idx):
    eid, ref, hol = pd.Series(0, idx), pd.Series(0, idx), pd.Series(np.nan, idx, dtype=object)
    for e, k in EIDS:
        for t in range(EID_WIN[0], EID_WIN[1] + 1):
            day = e + pd.Timedelta(days=t)
            if day in idx: eid[day] = 1; hol[day] = e.date().isoformat()
        for lo, hi in REF:
            for t in range(lo, hi + 1):
                day = e + pd.Timedelta(days=t)
                if day in idx: ref[day] = 1; hol[day] = e.date().isoformat()
    return eid, ref, hol
df["eid"], df["ref"], df["holiday"] = flags(df.index)
win = df[(df["eid"] == 1) | (df["ref"] == 1)].dropna(subset=["log_pm25"]).copy()
win = win.dropna(subset=wx_cov2)  # need weather for adjusted model

out = {}
out["weather_coverage_2016_2020"] = float(wx_coverage)
out["n_holidays"] = len(EIDS)
out["n_window_days"] = int(len(win))

# ---------------------------------------------------------------- (0) raw effect, same 8 holidays
def pooled_ratio(sub):
    lrs = []
    for e, k in EIDS:
        m = sub[sub["holiday"] == e.date().isoformat()]
        a = m[m["eid"] == 1]["pm25"]; b = m[m["ref"] == 1]["pm25"]
        if len(a) >= 4 and len(b) >= 14:
            lrs.append(np.log(a.mean() / b.mean()))
    lrs = np.array(lrs)
    boots = np.array([RNG.choice(lrs, len(lrs), replace=True).mean() for _ in range(10000)])
    return dict(n=len(lrs), pct=100*(np.exp(lrs.mean())-1),
                ci=[100*(np.exp(np.percentile(boots,2.5))-1), 100*(np.exp(np.percentile(boots,97.5))-1)],
                p=float(stats.ttest_1samp(lrs,0).pvalue))
out["raw_8holidays"] = pooled_ratio(win)

# ---------------------------------------------------------------- (1) covariate regression (OLS with holiday fixed effects)
import numpy as np
def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return beta, se, n, k

# design: intercept absorbed by holiday dummies; Eid dummy; weather covariates (standardized)
hols = sorted(win["holiday"].unique())
Xcols = {}
for h in hols[1:]:
    Xcols[f"hol_{h}"] = (win["holiday"] == h).astype(float).values
Xcols["eid"] = win["eid"].astype(float).values
Z = win[wx_cov2].copy()
Zs = (Z - Z.mean()) / Z.std()
for c in wx_cov2:
    Xcols[f"z_{c}"] = Zs[c].values
Xcols["_const"] = np.ones(len(win))
Xmat = np.column_stack(list(Xcols.values()))
y = win["log_pm25"].values
names_ = list(Xcols.keys())
beta, se, n, k = ols(y, Xmat)
i_eid = names_.index("eid")
b_eid, se_eid = beta[i_eid], se[i_eid]
tval = b_eid / se_eid
pval = 2 * stats.t.sf(abs(tval), n - k)
out["adjusted_regression"] = dict(
    pct=100*(np.exp(b_eid)-1),
    ci=[100*(np.exp(b_eid - 1.96*se_eid)-1), 100*(np.exp(b_eid + 1.96*se_eid)-1)],
    p=float(pval), n=int(n), covariates=wx_cov2,
    weather_betas={c: float(beta[names_.index(f"z_{c}")]) for c in wx_cov2},
)
# unadjusted version of the SAME regression (drop weather) for apples-to-apples
Xcols2 = {k2: v for k2, v in Xcols.items() if not k2.startswith("z_")}
Xmat2 = np.column_stack(list(Xcols2.values()))
b2, se2, n2, k2b = ols(y, Xmat2)
i2 = list(Xcols2.keys()).index("eid")
out["unadjusted_regression"] = dict(pct=100*(np.exp(b2[i2])-1),
    ci=[100*(np.exp(b2[i2]-1.96*se2[i2])-1), 100*(np.exp(b2[i2]+1.96*se2[i2])-1)], p=float(2*stats.t.sf(abs(b2[i2]/se2[i2]), n2-k2b)))

# ---------------------------------------------------------------- (2) deweathering (RandomForest counterfactual)
# train on REFERENCE days only, predict Eid-window days; effect = residual ratio
train = win[win["ref"] == 1]
feat = wx_cov2 + ["doy_sin","doy_cos"]
for d in (win, train):
    doy = d.index.dayofyear
    d["doy_sin"] = np.sin(2*np.pi*doy/365.25); d["doy_cos"] = np.cos(2*np.pi*doy/365.25)
rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf.fit(train[feat].values, train["log_pm25"].values)
win["pred_log"] = rf.predict(win[feat].values)
win["resid"] = win["log_pm25"] - win["pred_log"]
# pooled residual difference (Eid minus reference), per holiday, in %
lrs = []
for e, k in EIDS:
    m = win[win["holiday"] == e.date().isoformat()]
    a = m[m["eid"] == 1]["resid"]; b = m[m["ref"] == 1]["resid"]
    if len(a) >= 4 and len(b) >= 14:
        lrs.append(a.mean() - b.mean())
lrs = np.array(lrs)
boots = np.array([RNG.choice(lrs, len(lrs), replace=True).mean() for _ in range(10000)])
out["deweathered_rf"] = dict(n=len(lrs), pct=100*(np.exp(lrs.mean())-1),
    ci=[100*(np.exp(np.percentile(boots,2.5))-1), 100*(np.exp(np.percentile(boots,97.5))-1)],
    p=float(stats.ttest_1samp(lrs,0).pvalue),
    feature_importance={f: float(v) for f, v in zip(feat, rf.feature_importances_)})

# weather balance: are Eid windows systematically wetter/windier than reference?
bal = {}
for c in ["precip_mm","wind_ms","temp_c","rh_pct"]:
    a = win[win["eid"] == 1][c]; b = win[win["ref"] == 1][c]
    bal[c] = dict(eid_mean=float(a.mean()), ref_mean=float(b.mean()),
                  mwu_p=float(stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue))
out["weather_balance"] = bal

with open(f"{W}/weather_control.json","w") as fp:
    json.dump(out, fp, indent=2, default=float)
win[["pm25","log_pm25","pred_log","resid","eid","ref","holiday"]+wx_cov2].to_csv(f"{W}/weather_merged.csv")
print(json.dumps(out, indent=1, default=lambda x: round(float(x),4)))
