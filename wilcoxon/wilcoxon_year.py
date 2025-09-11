import numpy as np
from scipy.stats import wilcoxon

# 75.05
LargeScaleNet=np.array([65.527, 65.605, 66.096, 66.375, 65.479,
65.186, 65.685, 66.372, 65.366, 65.654,
64.537, 65.404, 66.233, 66.240, 65.489,
66.124, 65.602, 65.697, 65.683, 64.343,
65.758, 65.276, 65.434, 65.605, 64.721,
64.213, 65.059, 66.129, 64.877, 64.912])

FaberNet_1=np.array([
                   ])

FaberNet=np.array([
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
    print(f'  p-value: {p_value:.10f}\n')