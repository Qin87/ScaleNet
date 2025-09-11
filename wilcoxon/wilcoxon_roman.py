import numpy as np
from scipy.stats import wilcoxon

# 93.58
LargeScaleNet=np.array([93.735, 93.717, 93.505, 93.099, 93.329, 93.858, 93.717, 93.346, 93.823, 93.717,
                        94.070, 93.346, 93.346, 93.240, 93.346, 92.852, 93.805, 93.329, 93.787, 93.540,
                        94.140, 93.276, 93.117, 93.152, 93.540, 93.893, 93.717, 93.576, 93.311, 93.576])


FaberNet=np.array([93.134, 91.970, 91.899, 92.252, 92.252, 92.711, 92.146, 92.605, 92.640, 91.991,
                   92.076, 92.217, 92.093, 92.005, 92.676, 92.446, 92.252, 92.499, 92.411, 92.534,
                   92.640, 92.146, 92.111, 92.093, 92.270, 92.711, 92.252, 91.793, 92.693, 92.199])

FaberNet_1=np.array([
                   ])


models = [LargeScaleNet,FaberNet]
model_names = ['LargeScaleNet', 'FaberNet']

# Print mean and standard deviation for each model
for i, model in enumerate(models):
    mean = np.mean(model)
    std_dev = np.std(model)
    print(f'{model_names[i]}:{mean:.2f}±{std_dev:.2f}')

# List to store p-values
results = []

# Perform pairwise Wilcoxon signed-rank tests
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        stat, p_value = wilcoxon(models[i], models[j])
        results.append((f'{model_names[i]} vs {model_names[j]}', stat, p_value))

# Print Wilcoxon test results
for result in results:
    comparison, stat, p_value = result
    print(f'Comparison: {comparison}')
    print(f'  Statistic: {stat}')
    print(f'  p-value: {p_value:.12f}\n')

    # LargeScaleNet: 93.53±0.30
    # FaberNet: 92.32±0.30
    # Comparison: LargeScaleNet
    # vs
    # FaberNet
    # Statistic: 0.0
    # p - value: 0.0000000019