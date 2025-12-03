import matplotlib.pyplot as plt
dataset = 'Patents'

if dataset == 'WikiCS':
    models = ["MLP", "GCN-sp","SAGE-sp",
              "MagNet", "Dir-GNN", "FaberNet", "LScaleNet",
              "ScaleNet", "SAGE", "GCN", "APPNP", "Cheb"]

    gpu_512 = [182.25, 708.53, 845.96, 923.45, 1002.32,
              2961.4, 11641.43, 12222.3, 12249.13, 12255.33]
    time512= [0.0405, 0.1758, 0.0739, 0.1009, 0.13,
            0.6129, 0.7661, 0.5192, 9.3558, 0.7837]


    gpu_16 = [90.74, 383.9, 164.38, 193.85, 223.37,
             2832.5, 6841.75, 475.45, 476.17, 7187.47]
    time16 =[0.0417, 0.1134, 0.0492, 0.0576, 0.0677,
            0.2546, 0.3231, 0.0704, 0.3509, 0.3307]
elif dataset == 'Arxiv-year':
    models = ["MLP", "GCN-sp","SAGE-sp",
              "Dir-GNN", "FaberNet", "LScaleNet",
              "MagNet", "GCN", "APPNP", "SAGE", "Cheb"]

    # Your data reordered to same order
    gpu_1 = [205.1, 507.18, 675.22, 841.06, 1534.83, 740.34, 331.12, 12712.43, 13390.04]
    gpu_16 = [224.93, 804.13, 989.72, 1174.85, 1614.87, 1879.79, 1890.12, 12712.49, 13399.79]

    time1 = [0.4153, 0.6327, 0.4414, 0.4743, 0.4614, 0.5073, 1.1332, 0.9959, 1.0516]
    time16 = [0.4221, 0.5307, 0.5568, 0.6053, 1.2735, 0.5842, 1.818, 1.0641, 1.0733]

elif dataset == 'Chameleon':
    models = ["MLP", "GCN-sp","SAGE-sp",
              "Dir-GNN", "GCN",  "FaberNet",
              "LScaleNet", "ScaleNet", "MagNet", "SAGE", "Cheb", "APPNP"]
    gpu_16 = [87.27, 133.67, 134.04, 174.62,
                 215.87, 351.5, 360.2, 6597.75, 6929.96, 134.18]

    gpu_512 = [112.64, 277.56, 1617.25, 337.93,
                  409.41, 480.11, 527.35, 6633.04, 6977.5, 1625.67]


    time16 = [0.0181, 0.024, 0.0248, 0.0308,
                  0.0377, 0.0724, 0.0706, 0.2758, 0.2844, 0.0681]

    time512 = [0.0186, 0.03, 0.0816, 0.0413,
                   0.0531, 0.0934, 0.0856, 0.3329, 0.3423, 1.1961]

elif dataset == 'CiteSeer':
    models = ["MLP","GCN-sp","SAGE-sp", "Dir-GNN", "GCN", "FaberNet",
              "ScaleNet", "LScaleNet", "MagNet", "SAGE", "Cheb", "APPNP"]

    gpu_512 = [153.14, 160.28, 223.61,
               421.23, 463.56, 545.4,
                  676.19, 684.87, 842.28, 2380.14, 2472.52, 474.01]

    gpu_16 = [113.39, 113.08,161.19,
    212.46, 122.35, 306.89,
                 405.71, 401.78, 819.52, 2323.53, 2402.16, 208.89]

    time512 = [0.0221, 0.0359,0.0293,
               0.0372, 0.0418, 0.0524,
                   0.0729, 0.0679, 0.1116, 0.144, 0.1493, 0.3361]

    time16 = [0.0211, 0.0361,0.0257,
    0.0285, 0.0274, 0.036,
                  0.0512, 0.0436, 0.0901, 0.1272, 0.1323, 0.0566]
elif dataset == 'CoraML':
    models = ["MLP", "GCN-sp","SAGE-sp",
              "GCN", "Dir-GNN", "FaberNet",
              "LScaleNet", "ScaleNet", "MagNet", "SAGE", "Cheb", "APPNP"]

    # GPU memory (MB)
    gpu_16 = [99.37, 99.43, 133.15,
              112.68, 170.42, 236.93,
                 303.8, 316.88, 624.89, 2522.97, 2649.36, 195.03]

    gpu_512 = [131.9, 138.79,182.52,
    581.06, 360.75, 450.4,
                  553.81, 558.42, 631.34, 2566.9, 2708.21, 591.49]

    # Runtime (seconds)
    time16 = [0.0201, 0.0352,0.024,
    0.0262, 0.0255, 0.0313,
                  0.038, 0.0512, 0.0773, 0.1307, 0.1371, 0.054]

    time512 = [0.0205, 0.0342,0.0263,
    0.0327, 0.0448, 0.0448,
                   0.057, 0.0688, 0.0935, 0.1509, 0.1585, 0.437]
elif dataset== 'Patents':
    models_raw = ["MLP",  "GCN-sp","SAGE-sp",
                  "Dir-GNN", "GCN",  "FaberNet",
                  "LScaleNet", "MagNet", "SAGE", "Cheb", "APPNP"]

    gpu_1_mem_raw = [3838.37, 5686.82,10119.21,
                     13210.28, 10656.59, 17943.47,
                    23944.27, "OOM", "OOM", "OOM", 5403.47]
    gpu_1_time_raw = [6.8086,  8.1944,8.2966,
    7.828, 7.3736,  8.6296,
                     9.336, "OOM", "OOM", "OOM", 8.3022]

    gpu_16_mem_raw = [4321.68,  6021.45,10119.33,
                      16822.8, 25087.28,23180.54,
                     29538.33, "OOM", "OOM", "OOM",  25476.61]
    gpu_16_time_raw = [7.038,  7.376,7.3728,
    8.2965, 8.3208,10.1527,
                      10.9071, "OOM", "OOM", "OOM", 26.07]
elif dataset == 'Roman-Empire':
    models = ["MLP", "GCN-sp","SAGE-sp",
              "MagNet", "Dir-GNN", "FaberNet", "LScaleNet",
              "ScaleNet", "SAGE", "GCN", "Cheb", "APPNP"]

    # GPU memory
    gpu_512 = [284.66, 1293.16, 1511.65, 1655.41, 1802.18,
                  1829.41, 2728.21, 2813.72, 2869.35, 2870.48]

    gpu_16 = [101.59, 553.49, 198.27, 253.25, 308.1,
                 334.15, 1584.63, 192.17, 1663.1, 202.87]

    # Runtime
    time512 = [0.0719, 0.226, 0.1147, 0.1418, 0.1736,
                   0.1734, 0.2683, 0.1983, 0.2839, 2.6002]

    time16 = [0.067, 0.1223, 0.0698, 0.0734, 0.0891,
                  0.0927, 0.1412, 0.0793, 0.1543, 0.1569]
elif dataset == 'Squirrel':

    models = [
        "MLP", "GCN-sp","SAGE-sp",
        "Dir-GNN", "FaberNet", "LScaleNet", "MagNet",
        "GCN", "ScaleNet", "SAGE", "Cheb", "APPNP"
    ]

    gpu_512 = [157.18, 537.09, 651.87, 776.81, 856.04, 8862.46, 3794.64, 33912.55, 35682.33, 8876.1]
    time512 = [0.0275, 0.0607, 0.0928, 0.1258, 0.2182, 0.3615, 0.4585, 1.65, 1.681, 6.5726]

    gpu_16 = [115.31, 224.74, 309.63, 394.79, 770.81, 394.51, 3737.55, 33880.55, 35632.25, 393.97]
    time16 = [0.0272, 0.0468, 0.068, 0.089, 0.2199, 0.0459, 0.3506, 1.3379, 1.3615, 0.2578]
elif dataset == 'Telegram':
    models = ["MLP", "GCN-sp","SAGE-sp",
              "Dir-GNN", "MagNet", "FaberNet", "LScaleNet", "ScaleNet", "SAGE", "GCN", "APPNP", "Cheb"]
    gpu_16 = [65.33, 66.26, 68.28, 66.31, 66.45, 72.41, 75.02, 75.75, 75.76, 75.76]
    time16 = [0.0146, 0.0175, 0.048, 0.0205, 0.024, 0.0262, 0.019, 0.0202, 0.0478, 0.0244]

    gpu_512 = [66.23, 81.7, 83.14, 85.19, 96.06, 105.34, 407.87, 426.02, 430.48, 434.48]
    time512 = [0.0138, 0.0171, 0.0474, 0.0201, 0.0234, 0.026, 0.0304, 0.0322, 0.3042, 0.0359]

else:
    pass
if dataset in ['Patents']:
    models = []
    gpu_1 = []
    time1 = []
    gpu_16 = []
    time16 = []
    for m, m1, t1, m16, t16 in zip(models_raw, gpu_1_mem_raw, gpu_1_time_raw, gpu_16_mem_raw, gpu_16_time_raw):
        if m1 == "OOM" or m16 == "OOM":
            continue  # skip OOM models
        models.append(m)
        gpu_1.append(m1)
        time1.append(t1)
        gpu_16.append(m16)
        time16.append(t16)

plt.figure(figsize=(10, 5))
if dataset in ['WikiCS', 'Chameleon', 'CiteSeer', 'CoraML', 'Roman-Empire', 'Squirrel', 'Telegram']:
    ax1 = plt.gca()
    ax1.plot(models, gpu_512, marker='o', label="GPU mem (gpu__dim 512)")
    ax1.plot(models, gpu_16, marker='o', label="GPU mem (gpu__dim 16)")
    ax1.set_ylabel("GPU Memory (MB)")

    ax2 = ax1.twinx()
    ax2.plot(models, time512, marker='X', linestyle='--', label="Runtime (gpu__dim 512)")
    ax2.plot(models, time16, marker='X', linestyle='--', label="Runtime (gpu__dim 16)")
    ax2.set_ylabel("Runtime (s)")
    plt.title(dataset + ": GPU Memory and Runtime for gpu__dim 512 vs 16")
elif dataset in ['Arxiv-year', 'Patents']:
    ax1 = plt.gca()
    ax1.plot(models, gpu_1, marker='o', label="GPU mem (gpu__dim 1)")
    ax1.plot(models, gpu_16, marker='o', label="GPU mem (gpu__dim 16)")
    ax1.set_ylabel("GPU Memory (MB)")

    ax2 = ax1.twinx()
    ax2.plot(models, time1, marker='X', linestyle='--', label="Runtime (gpu__dim 1)")
    ax2.plot(models, time16, marker='X', linestyle='--', label="Runtime (gpu__dim 16)")
    ax2.set_ylabel("Runtime (s)")
    plt.title(dataset + ": GPU Memory and Runtime for gpu__dim 1 vs 16")


plt.xticks(rotation=0)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.savefig(dataset+".png", bbox_inches='tight')
plt.show()
