import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
W="/home/work"
r=json.load(open(f"{W}/results.json")); ev=pd.read_csv(f"{W}/per_event_results.csv")
curve=pd.read_csv(f"{W}/event_study_curve.csv"); diur=pd.read_csv(f"{W}/diurnal_relative.csv",index_col=0)
placebo=np.load(f"{W}/placebo_means.npy")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
 "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
C_F,C_A,C_E,C_R="#1f6f8b","#c0504d","#f2c14e","#bbbbbb"

# ---------- Figure 1 stacked ----------
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(3.3,4.5),gridspec_kw={"height_ratios":[1,1.25]})
ax1.axvspan(-21,-8,color=C_R,alpha=.25,lw=0); ax1.axvspan(8,21,color=C_R,alpha=.25,lw=0); ax1.axvspan(-1,3,color=C_E,alpha=.35,lw=0)
ax1.fill_between(curve["t"],curve["lo"],curve["hi"],color=C_F,alpha=.18,lw=0)
ax1.plot(curve["t"],curve["mean_pct"],color=C_F,lw=1.6); ax1.axhline(100,color="k",lw=.7,ls="--")
ax1.set_xlim(-21,21); ax1.set_ylim(40,140); ax1.set_xticks([-21,-14,-7,0,7,14,21])
ax1.set_xlabel("Days from Eid (0 = Eid)"); ax1.set_ylabel("PM$_{2.5}$ (% of\nreference)")
ax1.set_title("(a) Average of 16 Eid holidays",loc="left",fontsize=8.5)
inc=ev[ev["included"]].reset_index(drop=True)
inc["label"]=[f"{'F' if k=='Fitr' else 'A'} {pd.Timestamp(d).strftime('%b %y')}" for d,k in zip(inc["eid"],inc["kind"])]
y=np.arange(len(inc))[::-1]
for yi,x,k in zip(y,inc["pct_change"],inc["kind"]):
    c=C_F if k=="Fitr" else C_A; m="o" if k=="Fitr" else "s"
    ax2.plot([0,x],[yi,yi],color=c,lw=.8,alpha=.6); ax2.scatter([x],[yi],color=c,marker=m,s=14,zorder=3)
p=r["pooled_all"]
ax2.errorbar([p["pct"]],[-1.4],xerr=[[p["pct"]-p["ci_pct"][0]],[p["ci_pct"][1]-p["pct"]]],fmt="D",color="k",ms=5,capsize=2.5,lw=1,zorder=4)
ax2.axvline(0,color="k",lw=.7,ls="--")
ax2.set_yticks(list(y)+[-1.4]); ax2.set_yticklabels(list(inc["label"])+["Pooled"],fontsize=6.3)
ax2.set_xlabel("Change during Eid (%)"); ax2.set_xlim(-80,70)
ax2.set_title("(b) Each holiday and the pooled estimate",loc="left",fontsize=8.5)
ax2.legend(handles=[Line2D([],[],marker="o",color=C_F,ls="",label="Eid al-Fitr"),Line2D([],[],marker="s",color=C_A,ls="",label="Eid al-Adha")],loc="lower left",frameon=False,fontsize=6.3)
ax2.spines["left"].set_visible(False); ax2.tick_params(axis="y",length=0)
fig.tight_layout(h_pad=1.4); fig.savefig(f"{W}/Figure1_col.png",dpi=300); plt.close(fig)

# ---------- Figure 2 stacked ----------
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(3.3,4.3))
h=diur.index.values
ax1.plot(h,diur["ref"],color="#444",lw=1.5,label="Reference days"); ax1.plot(h,diur["eid"],color=C_F,lw=1.5,label="Eid-window days")
ax1.fill_between(h,diur["eid"],diur["ref"],color=C_E,alpha=.35,lw=0)
ax1.set_xlim(0,23); ax1.set_ylim(0,165); ax1.set_xticks([0,6,12,18,23])
ax1.set_xlabel("Hour of day (local time)"); ax1.set_ylabel("PM$_{2.5}$ (% of\nreference)")
ax1.set_title("(a) Hour-by-hour profile",loc="left",fontsize=8.5); ax1.legend(loc="upper center",frameon=False,ncol=2,fontsize=6.5)
pp=100*(np.exp(placebo)-1)
ax2.hist(pp,bins=40,color="#999",edgecolor="white",lw=.3); ax2.axvline(p["pct"],color=C_A,lw=1.8)
ax2.text(p["pct"]+1.5,ax2.get_ylim()[1]*.9,f"observed\n{p['pct']:.0f}%",color=C_A,fontsize=6.8,va="top")
ax2.set_xlabel("Pooled effect for random dates (%)"); ax2.set_ylabel("Count")
ax2.set_title("(b) Placebo test (2,000 draws)",loc="left",fontsize=8.5); ax2.set_xlim(-45,25)
fig.tight_layout(h_pad=1.4); fig.savefig(f"{W}/Figure2_col.png",dpi=300); plt.close(fig)
print("column figures done")
