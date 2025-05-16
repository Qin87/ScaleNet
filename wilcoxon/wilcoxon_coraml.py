import numpy as np
from scipy.stats import wilcoxon

# These arrays should contain the paired data
Scale = np.array([83.10, 80.47, 82.21, 80.51, 83.23, 82.21, 83.10, 83.69, 82.17, 82.51, 83.78, 80.93, 81.49, 81.57, 82.89, 83.01, 81.10, 81.57, 81.49, 84.25, 82.89, 81.53, 81.66, 81.10, 82.93, 82.76, 83.10, 83.65, 78.98, 82.72])
QiGu2 = np.array([82.89, 82.17, 78.77, 83.52, 83.91, 81.61, 81.06, 83.91, 81.02, 84.42, 83.61, 81.78, 80.55, 81.95, 84.08, 83.06, 80.93, 81.36, 80.72, 84.33, 84.16, 83.01, 81.66, 82.34, 84.84, 81.57, 80.68, 84.46, 81.32, 83.14])
QiG = np.array([83.14, 78.81, 80.21, 81.44, 83.78, 81.83, 82.12, 81.44, 80.81, 84.08, 83.35, 82.17, 80.93, 81.19, 82.25, 81.78, 81.57, 81.95, 80.25, 84.03, 83.52, 79.02, 82.21, 81.87, 82.42, 81.91, 81.15, 81.06, 82.34, 82.93])
Sym=np.array([78.81, 79.83, 81.40, 79.66, 83.52, 77.41, 76.86, 82.25, 82.25, 83.10, 81.40, 78.64, 80.42, 80.25, 82.85, 80.93, 80.85, 84.37, 80.00, 82.21, 79.92, 79.62, 80.04, 79.02, 82.51, 78.81, 79.58, 82.59, 79.92, 83.57])
Qym=np.array([81.32, 81.15, 80.08, 78.39, 80.38, 79.28, 81.44, 83.91, 79.62, 81.95, 80.04, 79.36, 80.51, 82.34, 82.17, 79.87, 82.04, 81.53, 80.13, 82.21, 80.17, 80.85, 79.32, 79.70, 82.00, 79.32, 79.36, 83.01, 80.98, 84.59])

models = [Scale,QiGu2, QiG, Sym, Qym]
model_names = ['Scale', 'QiGu2', 'QiG', 'Sym', 'Qym']

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