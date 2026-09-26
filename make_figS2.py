import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W="/home/work"; o=json.load(open(f"{W}/weather_control.json"))
main=json.load(open(f"{W}/results.json"))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
rows=[("All 16 holidays (main result)", main["pooled_all"]["pct"], main["pooled_all"]["ci_pct"], "#888"),
      ("8 holidays 2016-2020 (raw)", o["raw_8holidays"]["pct"], o["raw_8holidays"]["ci"], "#888"),
      ("Weather-adjusted regression", o["adjusted_regression"]["pct"], o["adjusted_regression"]["ci"], "#1f6f8b"),
      ("Deweathered (random forest)", o["deweathered_rf"]["pct"], o["deweathered_rf"]["ci"], "#1f6f8b")]
fig,ax=plt.subplots(figsize=(6.2,2.5)); y=np.arange(len(rows))[::-1]
for yi,(lab,p,ci,c) in zip(y,rows):
    ax.plot([ci[0],ci[1]],[yi,yi],color=c,lw=2.2,solid_capstyle="round")
    ax.plot([p],[yi],"D",color=c,ms=7)
    ax.text(p, yi+0.18, f"{p:.0f}%", ha="center", fontsize=8, color=c)
ax.axvline(0,color="k",lw=0.8,ls="--")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlabel("Change in PM$_{2.5}$ during Eid (%)"); ax.set_xlim(-70,5)
ax.spines["left"].set_visible(False); ax.tick_params(axis="y",length=0)
ax.set_title("Weather adjustment does not remove the Eid effect", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(f"{W}/FigureS2.png", dpi=300); print("S2 done")
