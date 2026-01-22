import re
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# -----------------------------
# 1) Paste your lines here
# -----------------------------
RAW = r"""
78.86±1.47_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.005_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_044635.log
78.86±1.47_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.01_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_043723.log
78.86±1.47_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.04_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_041837.log
78.97±1.94_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-07_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_061647.log
79.17±1.91_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg150.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_011735.log
79.25±1.73_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.5_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_033122.log
79.25±1.73_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg1.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_032137.log
79.25±1.73_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg16.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_024511.log
79.25±1.73_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg2.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_031239.log
79.25±1.73_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg32.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_022606.log
79.25±1.73_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg4.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_030337.log
79.25±1.73_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg8.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_025429.log
79.30±1.69_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg140.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_012713.log
79.34±1.61_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg120.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_013745.log
79.34±1.68_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.3_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_035013.log
79.39±1.25_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.03_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_042726.log
79.41±1.41_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.0005_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_050639.log
79.41±1.41_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.001_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_045607.log
79.43±1.54_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.05_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_004909.log
79.43±1.54_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.1_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_040849.log
79.43±1.54_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.2_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_035934.log
79.43±1.54_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg0.4_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_034055.log
79.43±1.54_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Bal_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260102_223115.log
79.45±1.48_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg9e-07_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_060701.log
79.45±1.63_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-05_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_051618.log
79.45±1.69_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg160.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_010746.log
79.45±1.69_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg20.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_023524.log
79.45±1.69_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg40.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_021619.log
79.50±1.74_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-08_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_062603.log
79.50±2.13_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg2e-06_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_054649.log
79.52±1.84_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-06_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_052646.log
79.63±1.23_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg3e-06_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_053629.log
79.71±1.68_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg100.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_015742.log
79.71±1.68_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg200.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_005829.log
79.71±1.68_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg50.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_020700.log
79.71±1.68_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Bal_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260102_223544.log
79.76±1.52_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg110.0_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_014739.log
79.89±1.25_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-08_s101_dp0.0_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260121_220110.log
79.91±1.86_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg1e-06_s101_dp0.5_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260103_055644.log
80.00±1.25_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg10000.0_s101_dp0.0_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260121_215357.log
80.20±1.58_main_WikipediaNetwork_chameleonA_NoBN_Dir-GNN_NoSloop_hid128_Balcoef_agg1000.0_s101_dp0.0_split10_n1_lay5_lr0.005_Sch_P400_NoImp810_20260121_215014.log


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
plt.savefig("chameleon_accuracy_vs_coef_n1.pdf", format="pdf", bbox_inches="tight",  pad_inches=0.05,)
plt.show()
