import numpy as np
from scipy.stats import wilcoxon

# These arrays should contain the paired data
ScaleNet=np.array([66.31, 67.68, 69.39, 68.35, 69.87, 67.98, 69.58, 67.05, 70.28, 66.20, 66.05, 68.91, 68.31, 67.01, 69.58, 67.79, 69.28, 65.23, 70.02, 65.86, 67.42, 69.80, 69.95, 68.46, 70.02, 67.83, 68.68, 66.31, 68.46, 66.27])
QiGi2=np.array([64.26, 69.09, 70.88, 67.57, 69.95, 68.57, 67.31, 65.42, 69.84, 66.01, 61.40, 66.68, 68.20, 66.72, 68.83, 68.42, 64.64, 65.79, 67.79, 65.71, 62.89, 67.50, 67.79, 66.98, 68.65, 68.87, 64.19, 65.86, 67.35, 66.23])
Sym=np.array([62.33, 63.26, 66.94, 67.12, 68.65, 64.60, 67.27, 66.49, 67.42, 64.15, 60.62, 66.49, 66.75, 62.82, 69.02, 66.12, 64.38, 60.48, 65.53, 61.74, 64.30, 65.71, 68.61, 64.04, 69.47, 68.39,
               66.38, 64.15, 67.31, 62.74])
QiGu2=np.array([62.89, 66.42, 66.60, 63.26, 65.68, 65.30, 64.90, 66.79, 63.60, 65.30, 64.78, 65.97, 68.54, 67.46, 69.65, 65.34, 65.56, 66.01, 67.98, 65.38, 66.08, 66.79, 68.76, 67.12, 68.20, 68.68, 65.75, 66.68, 66.57, 64.56])

models = [ScaleNet,QiGi2, Sym,  QiGu2]
model_names = ['ScaleNet', 'QiGi2',  'Sym', 'QiGu2']

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