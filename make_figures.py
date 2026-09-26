import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

W = "/home/work"
r = json.load(open(f"{W}/results.json"))
ev = pd.read_csv(f"{W}/per_event_results.csv")
curve = pd.read_csv(f"{W}/event_study_curve.csv")
diur = pd.read_csv(f"{W}/diurnal_relative.csv", index_col=0)
placebo = np.load(f"{W}/placebo_means.npy")
daily = pd.read_csv(f"{W}/daily_pm25_dhaka_embassy.csv", index_col=0, parse_dates=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})
C_FITR, C_ADHA, C_EID, C_REF = "#1f6f8b", "#c0504d", "#f2c14e", "#bbbbbb"

# ------------------------------------------------------------------ Figure 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
# (a) event-study curve
ax1.axvspan(-21, -8, color=C_REF, alpha=0.25, lw=0)
ax1.axvspan(8, 21, color=C_REF, alpha=0.25, lw=0)
ax1.axvspan(-1, 3, color=C_EID, alpha=0.35, lw=0)
ax1.fill_between(curve["t"], curve["lo"], curve["hi"], color=C_FITR, alpha=0.18, lw=0)
ax1.plot(curve["t"], curve["mean_pct"], color=C_FITR, lw=1.8)
ax1.axhline(100, color="k", lw=0.8, ls="--")
ax1.set_xlim(-21, 21); ax1.set_ylim(40, 140)
ax1.set_xlabel("Days relative to Eid day (0 = Eid)")
ax1.set_ylabel("Daily PM$_{2.5}$ (% of each Eid's\nown reference level)")
ax1.set_xticks([-21, -14, -7, 0, 7, 14, 21])
ax1.text(-14.5, 133, "reference\nwindow", ha="center", va="top", fontsize=7.5, color="#555")
ax1.text(14.5, 133, "reference\nwindow", ha="center", va="top", fontsize=7.5, color="#555")
ax1.text(1, 133, "Eid\nwindow", ha="center", va="top", fontsize=7.5, color="#7a5c00")
ax1.set_title("(a) Average of 16 Eid holidays, 2016-2024", loc="left")
ax1.text(0.98, 0.04, "band = 95% bootstrap CI", transform=ax1.transAxes, fontsize=7, color="#555", ha="right")

# (b) forest plot
inc = ev[ev["included"]].copy().reset_index(drop=True)
inc["label"] = [f"{k} {pd.Timestamp(d).strftime('%b %Y')}" for d, k in zip(inc["eid"], inc["kind"])]
y = np.arange(len(inc))[::-1]
colors = [C_FITR if k == "Fitr" else C_ADHA for k in inc["kind"]]
markers = ["o" if k == "Fitr" else "s" for k in inc["kind"]]
for yi, x, c, m in zip(y, inc["pct_change"], colors, markers):
    ax2.plot([0, x], [yi, yi], color=c, lw=1, alpha=0.6)
    ax2.scatter([x], [yi], color=c, marker=m, s=22, zorder=3)
p = r["pooled_all"]
ax2.errorbar([p["pct"]], [-1.3], xerr=[[p["pct"] - p["ci_pct"][0]], [p["ci_pct"][1] - p["pct"]]],
             fmt="D", color="k", ms=6, capsize=3, lw=1.2, zorder=4)
ax2.axvline(0, color="k", lw=0.8, ls="--")
ax2.axvline(p["pct"], color="k", lw=0.6, ls=":")
ax2.set_yticks(list(y) + [-1.3]); ax2.set_yticklabels(list(inc["label"]) + ["Pooled (95% CI)"], fontsize=7)
ax2.set_xlabel("Change in PM$_{2.5}$ during Eid window (%)")
ax2.set_xlim(-80, 70)
ax2.set_title("(b) Each holiday; pooled estimate", loc="left")
ax2.legend(handles=[Line2D([], [], marker="o", color=C_FITR, ls="", label="Eid al-Fitr"),
                    Line2D([], [], marker="s", color=C_ADHA, ls="", label="Eid al-Adha")],
           loc="upper right", frameon=False)
ax2.spines["left"].set_visible(False); ax2.tick_params(axis="y", length=0)
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{W}/Figure1.png", dpi=300); fig.savefig(f"{W}/Figure1.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Figure 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
h = diur.index.values
ax1.plot(h, diur["ref"], color="#444", lw=1.8, label="Reference days (n = 429)")
ax1.plot(h, diur["eid"], color=C_FITR, lw=1.8, label="Eid-window days (n = 79)")
ax1.fill_between(h, diur["eid"], diur["ref"], color=C_EID, alpha=0.35, lw=0)
ax1.set_xlim(0, 23); ax1.set_ylim(0, 165)
ax1.set_xticks([0, 4, 8, 12, 16, 20, 23])
ax1.set_xlabel("Hour of day (local time)"); ax1.set_ylabel("Hourly PM$_{2.5}$ (% of each Eid's\nreference level)")
ax1.set_title("(a) Hour-by-hour profile", loc="left")
ax1.legend(loc="upper right", frameon=False)
d = r["diurnal"]
ax1.text(0.02, 0.06, f"Reduction: night (00-05h) {d['night_0_5_reduction_pct']:.0f}%,  "
         f"morning (07-10h) {d['morning_7_10_reduction_pct']:.0f}%,\nafternoon (12-16h) "
         f"{d['afternoon_12_16_reduction_pct']:.0f}%,  evening (20-23h) {d['evening_20_23_reduction_pct']:.0f}%",
         transform=ax1.transAxes, fontsize=7, color="#333")

pp = 100 * (np.exp(placebo) - 1)
ax2.hist(pp, bins=40, color="#999", edgecolor="white", lw=0.4)
ax2.axvline(p["pct"], color=C_ADHA, lw=2)
ax2.text(p["pct"] + 1.5, ax2.get_ylim()[1] * 0.9, f"observed Eid\neffect {p['pct']:.0f}%", color=C_ADHA,
         fontsize=7.5, va="top")
ax2.set_xlabel("Pooled effect for 16 random non-Eid dates (%)")
ax2.set_ylabel("Count (2,000 placebo draws)")
ax2.set_title("(b) Placebo test", loc="left")
ax2.set_xlim(min(p["pct"] - 8, pp.min() - 3), pp.max() + 3)
pl = r["placebo"]
ax2.text(0.98, 0.72, f"placebo mean {pl['placebo_mean_pct']:.1f}%\n95% range {pl['placebo_pct_2_5']:.0f}% to "
         f"{pl['placebo_pct_97_5']:.0f}%\nmost extreme {pl['min_placebo_pct']:.0f}%\np < 0.0005",
         transform=ax2.transAxes, fontsize=7, ha="right", va="top", color="#333")
fig.tight_layout(w_pad=1.5)
fig.savefig(f"{W}/Figure2.png", dpi=300); fig.savefig(f"{W}/Figure2.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Supplementary Figure S1: small multiples
fig, axes = plt.subplots(4, 4, figsize=(7.5, 7.0), sharex=True)
for ax, row in zip(axes.flat, inc.itertuples()):
    e = pd.Timestamp(row.eid)
    days = pd.date_range(e - pd.Timedelta(days=21), e + pd.Timedelta(days=21))
    vals = daily["pm25"].reindex(days).values
    t = np.arange(-21, 22)
    ax.axvspan(-1, 3, color=C_EID, alpha=0.4, lw=0)
    ax.axvspan(-21, -8, color=C_REF, alpha=0.25, lw=0); ax.axvspan(8, 21, color=C_REF, alpha=0.25, lw=0)
    ax.plot(t, vals, color=C_FITR if row.kind == "Fitr" else C_ADHA, lw=1.2, marker="o", ms=1.8)
    ax.axhline(row.ref_mean, color="k", lw=0.7, ls="--")
    ax.set_title(f"{row.kind} {e.strftime('%d %b %Y')}   {row.pct_change:+.0f}%", fontsize=7.5, loc="left")
    ax.set_xlim(-21, 21); ax.set_xticks([-21, -7, 0, 7, 21]); ax.tick_params(labelsize=6.5)
    ax.set_ylim(0, max(np.nanmax(vals) * 1.1, 20))
for ax in axes[-1]:
    ax.set_xlabel("Days relative to Eid", fontsize=7.5)
for ax in axes[:, 0]:
    ax.set_ylabel("PM$_{2.5}$ (µg/m³)", fontsize=7.5)
fig.suptitle("Daily mean PM$_{2.5}$ around each analysed Eid (dashed = reference mean; yellow = Eid window; grey = reference windows)", fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(f"{W}/FigureS1.png", dpi=300)
plt.close(fig)
print("figures written")
