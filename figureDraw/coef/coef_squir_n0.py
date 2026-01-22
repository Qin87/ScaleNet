import re
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt

# -----------------------------
# 1) Paste your lines here
# -----------------------------
RAW = r"""
19.49±1.21_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1e-09_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_083024.log
 19.49±1.21_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1e-10_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_084154.log
 19.53±0.70_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg300000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_002255.log
 19.58±0.72_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg30000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_001439.log
 19.70±0.69_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg100000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_000911.log
 19.70±0.69_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg200000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_231025.log
 19.80±0.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg20000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_230250.log
 19.84±1.28_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-09_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_083602.log
 19.88±1.34_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-08_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_071723.log
 19.88±1.40_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-08_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_223124.log
 20.27±0.81_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg10000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_000037.log
'20.5±0.9_ 3splits_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg10000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_230057.log'
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1000000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_235434.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg100000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_002747.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg10000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_002144.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_001540.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2000000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_225648.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg200000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_232842.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg20000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_232241.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_231640.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg300000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_004403.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg30000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_003656.log
 20.83±1.44_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg3000000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_002956.log
 24.57±2.53_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_223837.log
 25.73±4.24_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg120.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_005610.log
 26.10±4.53_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg160.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_000233.log
 26.17±4.95_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg360.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_024402.log
 26.25±3.62_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg200.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_234447.log
 26.52±2.83_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg600.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_010054.log
 26.80±2.87_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg48.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_055244.log
 26.90±3.93_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg100.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_013321.log
 27.10±4.08_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg150.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_041213.log
 27.26±3.56_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg120.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_043404.log
 27.36±4.72_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg140.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_003716.log
 27.36±4.95_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg150.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_001710.log
 27.58±4.20_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg110.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_011206.log
 27.69±5.42_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg32.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_005631.log
 27.83±5.64_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg480.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_012138.log
 28.14±3.59_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg450.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_015034.log
 28.28±6.61_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg16.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_013955.log
 28.72±6.72_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg40.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_003358.log
 28.75±3.71_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg96.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_045937.log
 28.79±3.58_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg330.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_031051.log
 29.49±2.97_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg420.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_021618.log
 29.72±9.03_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg60.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_052335.log
 30.05±5.91_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1.5e-07_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_230825.log
 30.19±7.43_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg50.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_015005.log
 30.63±5.81_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg3000.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_234303.log
 30.85±7.61_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg300.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_033639.log
 31.56±7.06_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg20.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_011408.log
 32.06±6.86_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg12.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_064407.log
 32.59±6.73_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg8.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_015533.log
 33.20±8.98_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg24.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_061046.log
 35.43±5.68_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg6.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_070539.log
 38.68±3.04_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg3.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_073026.log
 40.57±4.71_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1.5_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_075207.log
 41.08±3.24_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg4.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_021406.log
 41.82±3.75_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_024045.log
 43.26±5.33_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1.0_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_025646.log
 45.20±10.32_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-07_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_062645.log
 47.94±5.23_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1.2000000000000002_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_080754.log
 48.05±7.83_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.8999999999999999_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_084039.log
 56.70±6.76_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.5_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_030659.log
 57.07±7.46_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg9e-07_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_051818.log
 60.03±1.17_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg1e-06_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_040426.log
 61.21±7.86_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.6000000000000001_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_090323.log
 64.90±2.12_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg2e-06_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_025108.log
 65.43±2.17_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg3e-06_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_014541.log
 65.77±1.91_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-06_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_005049.log
 68.99±1.95_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg5e-05_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_001850.log
 69.02±3.74_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.4_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_031942.log
 73.20±1.79_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.0005_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_235702.log
 74.14±1.30_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.001_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_233842.log
 74.14±1.82_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.30000000000000004_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_092609.log
 74.61±1.88_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.3_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_033613.log
 74.89±1.51_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.15000000000000002_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_005103.log
 74.92±1.37_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.005_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_232522.log
 74.95±1.72_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.2_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_035048.log
 74.96±1.71_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.1_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_040135.log
 75.03±1.42_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.01_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_231421.log
 75.16±1.70_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.12_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_094404.log
 75.21±1.15_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.05_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_233445.log
 75.21±1.49_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.03_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260121_230402.log
 75.22±1.42_main_WikipediaNetwork_squirrelA_BNorm_Dir-GNN_NoSloop_hid128_Balcoef_agg0.04_s11_dp0.0_split10_n0_lay6_lr0.01_Sch_P400_NoImp810_20260122_041118.log
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
plt.savefig("squirrel_accuracy_vs_coef_n0.pdf", format="pdf", bbox_inches="tight",  pad_inches=0.05,)
plt.show()
