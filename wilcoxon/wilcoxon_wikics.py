import numpy as np
from scipy.stats import wilcoxon

# These arrays should contain the paired data
Scale = np.array([78.01, 76.96, 78.88, 77.85, 78.62, 78.30, 77.41, 77.36, 78.54, 77.97, 80.09, 78.26, 80.01, 78.86, 79.03, 79.44, 78.93, 78.91, 79.82, 79.34, 78.31, 78.40, 79.32, 77.24, 77.90, 78.02, 77.95, 76.45, 78.91, 77.54])
QiGi2 = np.array([80.37, 78.67, 79.65, 79.1, 79.77, 79.66, 78.67, 78.19, 79.31, 79.31, 79.44, 78.72, 79.19, 78.38, 79.61, 79.55, 79.08, 78.19, 79.34, 79.32, 79.89, 78.90, 79.12, 78.93, 80.01, 79.51,
              79.07, 79.56, 79.00, 79.20])
DiGib = np.array([79.19, 79.01, 79.13, 78.04, 79.55, 79.08, 77.39, 78.25, 78.31, 78.26, 78.43, 79.39, 78.78, 77.60, 79.12, 78.74, 77.53, 77.72, 78.57, 78.04, 78.71, 79.03, 78.88, 77.78, 79.63, 79.56,
              77.42, 78.31, 78.01, 78.09])
DirGNN = np.array([78.26, 76.02, 77.99, 76.40, 77.13, 77.03, 77.08, 77.51, 77.85, 76.74, 78.72, 76.23, 77.87, 76.57, 76.64, 77.56, 77.37, 77.10, 78.31, 76.35, 78.74, 76.57, 77.80, 76.65, 77.41, 77.92, 77.17, 77.15, 78.23, 77.03])
Cheb = np.array([76.21, 76.09, 76.86, 74.55, 75.83, 75.78, 75.46, 74.40, 75.73, 74.60, 76.14, 76.07, 77.29, 75.18, 76.52, 76.47, 75.17, 75.41, 75.92, 74.81, 75.65, 75.22, 75.61, 74.53, 75.87, 76.28,
              74.19, 75.85, 75.68, 76.60])

models = [Scale,QiGi2, DiGib, DirGNN, Cheb]
model_names = ['Scale', 'QiGi2', 'DiGib', 'DirGNN', 'Cheb']

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
    print(f'  p-value: {p_value:.4f}\n')