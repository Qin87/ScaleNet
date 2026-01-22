import re
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# -----------------------------
# 1) Paste your lines here
# -----------------------------
RAW = r"""
19.58±1.83_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg300000000.0_s101_dp0.0_split10_n0_lay5_lr0.005_Sch_P400_NoImp810_20260122_085850.log
19.58±1.83_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg30000000.0_s101_dp0.0_split10_n0_lay5_lr0.005_Sch_P400_NoImp810_20260122_085119.log
19.63±1.89_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg3000000.0_s101_dp0.0_split10_n0_lay5_lr0.005_Sch_P400_NoImp810_20260122_084346.log
19.69±1.83_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg300000.0_s101_dp0.0_split10_n0_lay5_lr0.005_Sch_P400_NoImp810_20260122_083455.log
19.71±1.82_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg150000.0_s101_dp0.0_split10_n0_lay5_lr0.005_Sch_P400_NoImp810_20260122_082456.log
22.39±2.41_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg5e-08_lay5_lr0.005_NoImp810_20260103090855.log
23.09±3.62_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg9e-07_lay5_lr0.005_NoImp810_20260103085859.log
23.09±3.71_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg1e-06_lay5_lr0.005_NoImp810_20260103085403.log
23.11±3.61_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg5e-07_lay5_lr0.005_NoImp810_20260103090355.log
24.32±3.89_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg2e-06_lay5_lr0.005_NoImp810_20260103084818.log
30.44±7.79_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg3e-06_lay5_lr0.005_NoImp810_20260103083707.log
34.50±9.23_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg5e-06_lay5_lr0.005_NoImp810_20260103082447.log
62.76±3.84_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg1000.0_s101_dp0.0_split10_n0_lay5_lr0.005_Sch_P400_NoImp810_20260121_220836.log
67.89±1.97_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg5e-05_lay5_lr0.005_NoImp810_20260103074749.log
68.75±3.77_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg32.0_lay5_lr0.005_NoImp810_20260103034012.log
69.23±3.06_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg140.0_lay5_lr0.005_NoImp810_20260103013647.log
69.25±2.33_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg10000.0_s101_dp0.0_split10_n0_lay5_lr0.005_Sch_P400_NoImp810_20260121_222020.log
69.56±3.80_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg40.0_lay5_lr0.005_NoImp810_20260103032031.log
70.07±3.65_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg120.0_lay5_lr0.005_NoImp810_20260103015558.log
70.46±2.71_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg150.0_lay5_lr0.005_NoImp810_20260103011437.log
70.46±3.00_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg100.0_lay5_lr0.005_NoImp810_20260103023938.log
70.61±3.55_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg20.0_lay5_lr0.005_NoImp810_20260103035735.log
70.79±2.24_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg160.0_lay5_lr0.005_NoImp810_20260103005142.log
71.27±3.39_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg110.0_lay5_lr0.005_NoImp810_20260103021652.log
71.34±3.30_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg50.0_lay5_lr0.005_NoImp810_20260103025952.log
71.47±3.23_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg200.0_lay5_lr0.005_NoImp810_20260103002751.log
71.85_main_cora_ml_A_NoBN_1iA_Head8_AddSloop_hid64_Bal_s0_dp0_split1_n0_lay1_lr0.1_NoSch_NoImp410_20260121_214301.log
72.89±1.82_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.0005_lay5_lr0.005_NoImp810_20260103072429.log
73.42±3.18_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg16.0_lay5_lr0.005_NoImp810_20260103041530.log
74.56±0.98_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.001_lay5_lr0.005_NoImp810_20260103070446.log
76.84±1.39_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.005_lay5_lr0.005_NoImp810_20260103064802.log
77.19±1.48_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg8.0_lay5_lr0.005_NoImp810_20260103043418.log
77.65±1.69_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.01_lay5_lr0.005_NoImp810_20260103063315.log
78.22±1.44_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.03_lay5_lr0.005_NoImp810_20260103061846.log
78.40±1.60_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.04_lay5_lr0.005_NoImp810_20260103060455.log
78.44±1.89_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.1_lay5_lr0.005_NoImp810_20260103055404.log
78.57±1.61_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.05_lay5_lr0.005_NoImp810_20260103001441.log
79.06±0.70_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg4.0_lay5_lr0.005_NoImp810_20260103045053.log
79.06±1.33_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.2_lay5_lr0.005_NoImp810_20260103054405.log
79.23±1.12_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg1.0_lay5_lr0.005_NoImp810_20260103050842.log
79.28±1.32_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.3_lay5_lr0.005_NoImp810_20260103053433.log
79.30±1.26_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.5_lay5_lr0.005_NoImp810_20260103051624.log
79.34±1.01_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg0.4_lay5_lr0.005_NoImp810_20260103052537.log
79.47±1.13_WikipediaNetwork_chameleonDirect_NoBNorm_Dir-GNN_NoSloop128hid__Balcoef_agg2.0_lay5_lr0.005_NoImp810_20260103050050.log


""".strip()

# -----------------------------
# 2) Parse lines: mean±std and agg value
# -----------------------------
@dataclass(frozen=True)
class Row:
    agg: float
    mean: float
    std: float
    name: str

# mean±std_..._agg<value>_...
PAT = re.compile(
    r"^(?P<mean>\d+(?:\.\d+)?)±(?P<std>\d+(?:\.\d+)?)_.*?_agg(?P<agg>[0-9]*\.?[0-9]+(?:e[+-]?\d+)?)_.*$",
    re.IGNORECASE,
)

rows: List[Row] = []
bad: List[str] = []

for line in RAW.splitlines():
    line = line.strip()
    if not line:
        continue
    m = PAT.match(line)
    if not m:
        bad.append(line)
        continue
    rows.append(
        Row(
            agg=float(m.group("agg")),
            mean=float(m.group("mean")),
            std=float(m.group("std")),
            name=line,
        )
    )

if bad:
    print("WARNING: could not parse these lines:")
    for b in bad:
        print("  ", b)

# sort by agg for plotting
rows.sort(key=lambda r: r.agg)

aggs = [r.agg for r in rows]
means = [r.mean for r in rows]
stds  = [r.std for r in rows]

# min/max by mean (ties -> first after sorting by agg)
min_i = min(range(len(rows)), key=lambda i: means[i])
max_i = max(range(len(rows)), key=lambda i: means[i])

print(f"MIN: agg={aggs[min_i]:g}, mean±std={means[min_i]:.2f}±{stds[min_i]:.2f}")
print(f"MAX: agg={aggs[max_i]:g}, mean±std={means[max_i]:.2f}±{stds[max_i]:.2f}")

# -----------------------------
# 3) Plot (error bars) + annotate min/max
# -----------------------------
plt.figure(figsize=(16, 6))
plt.errorbar(aggs, means, yerr=stds, fmt="o-", capsize=3,  color="green")

ymin, ymax = plt.ylim()
plt.ylim(ymin, ymax + 0.3*(ymax - ymin))  # +15% on top


# log scale is usually best for agg sweeps (covers 5e-08 ... 200)
plt.xscale("log")
plt.xlabel("Coefficient", fontsize=28)
plt.ylabel("Accuracy (%)", fontsize=28)
plt.title("Squirrel: Accuracy vs Coefficient (mean ± std)", fontsize=30)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# highlight min/max points (no custom colors; use markers)
plt.plot(aggs[min_i], means[min_i], marker="v", markersize=16, linestyle="None")
plt.plot(aggs[max_i], means[max_i], marker="^", markersize=16, linestyle="None")

# annotate
def annotate(i: int, label: str, x, y):
    plt.annotate(
        f"{label}\ncoef={aggs[i]:g}\n{means[i]:.2f}±{stds[i]:.2f}",
        xy=(aggs[i], means[i]),
        xytext=(x, y),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=26,
        ha="left",
        va="bottom",
    )

annotate(min_i, "MIN", -250, 120)
annotate(max_i, "MAX", -210, 90)

plt.tight_layout()
plt.savefig("chameleon_accuracy_vs_coef_n0.pdf", format="pdf", bbox_inches="tight",  pad_inches=0.05,)
plt.show()
