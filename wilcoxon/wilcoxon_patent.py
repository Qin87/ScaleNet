import numpy as np
from scipy.stats import wilcoxon

# 75.05
LargeScaleNet=np.array([75.164, 75.035, 75.013, 75.030, 74.915,
                         74.993, 75.128, 75.103, 75.002, 75.024,
                         75.094, 74.998, 74.843, 75.122, 74.912,
                        74.983, 75.015, 74.962, 75.106, 74.842,
                        75.053, 74.735, 74.684, 75.188, 74.747,
                        75.051, 74.829, 74.912, 74.980, 74.922])


FaberNet=np.array([74.518, 74.612, 74.484, 74.674, 74.649,
                   74.274, 74.669, 74.466, 74.588, 74.587,
                    74.514, 74.437, 74.312, 74.667, 74.508,
                   74.379, 74.566, 74.556, 74.562, 74.474,
                   74.518, 74.612, 74.484, 74.674, 74.649,
                   74.702, 74.702, 74.614, 74.525, 74.531
                   ])


# DirGNN=np.array([])


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
    print(f'  p-value: {p_value:.11f}\n')

    # Comparison: LargeScaleNet vs FaberNet
    #   Statistic: 0.0
    #   p-value: 0.0000000019