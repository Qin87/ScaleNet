import re
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# -----------------------------
# 1) Paste your lines here
# -----------------------------
RAW = r"""
74.87±1.98_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-08_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_210540.log
75.00±1.88_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.03_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_185136.log
75.21±1.81_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg200000000.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260121_234401.log
75.21±2.56_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2000000.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260121_232341.log
75.22±1.74_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg3e-06_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_200935.log
75.23±1.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.5_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_174900.log
75.23±1.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_173849.log
75.23±1.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg16.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_165809.log
75.23±1.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_172839.log
75.23±1.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg32.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_163458.log
75.23±1.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg4.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_171829.log
75.23±1.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg8.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_170819.log
75.24±1.76_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-07_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_205532.log
75.27±1.69_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1e-06_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_203304.log
75.27±2.18_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg200000.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260121_231355.log
75.29±1.82_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg110.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_154939.log
75.30±2.05_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg160.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_150143.log
75.30±2.05_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg20.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_164508.log
75.30±2.05_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg40.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_162209.log
75.37±1.98_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.005_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_191348.log
75.37±1.98_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.01_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_190303.log
75.37±1.98_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.04_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_184051.log
75.38±2.12_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-05_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_194826.log
75.38±2.69_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg20000.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260121_230424.log
75.39±2.24_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.3_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_180929.log
75.45±2.11_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1000000000.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260121_224134.log
75.45±2.11_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2000000000.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260121_225401.log
75.46±1.71_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-06_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_200003.log
75.47±1.48_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2e-06_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_202156.log
75.47±1.93_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg140.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_152649.log
75.49±2.03_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.05_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_144042.log
75.49±2.03_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.1_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_183032.log
75.49±2.03_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.2_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_182014.log
75.49±2.03_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.4_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_175910.log
75.54±1.78_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.0005_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_193629.log
75.54±1.78_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.001_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_192433.log
75.54±1.80_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg100.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_160043.log
75.54±1.80_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg200.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_145101.log
75.54±1.80_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg50.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_161125.log
75.55±2.02_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg20000000.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260121_233347.log
75.58±2.02_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg120.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_153740.log
75.59±1.59_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg9e-07_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_204415.log
75.65±1.92_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg150.0_s11_dp0.0_split10_n1_lay6_lr0.01_Sch_P400_NoImp810_20260103_151432.log
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

annotate(min_i, "MIN", 50, 120)
annotate(max_i, "MAX", -50, 90)

plt.tight_layout()
plt.savefig("squirrel_accuracy_vs_coef_n1.pdf", format="pdf", bbox_inches="tight",  pad_inches=0.05,)
plt.show()
