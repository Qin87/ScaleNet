import numpy as np
from scipy.stats import wilcoxon

# Sample data
# These arrays should contain the paired data
APPNP = np.array([78.70, 78.03, 77.02, 78.48, 78.43, 78.96, 77.91, 78.83, 77.15, 78.50, 77.04, 77.11, 78.47, 79.07, 78.72, 78.23, 78.33, 77.17, 76.81, 77.67, 78.90, 79.02, 78.88, 78.13, 78.46,
                  78.16, 77.82, 79.04, 78.83, 78.49])
Mag = np.array([])
Scale = np.array([76.05, 76.09, 77.76, 73.68, 74.12, 74.26, 76.33, 77.26, 76.26, 75.44, 73.49, 77.87, 77.66, 76.87, 76.84, 76.43, 73.50, 70.61, 72.71, 71.62, 74.88, 76.98, 78.37, 73.35, 74.88,
                   75.96,
              73.05, 73.34, 76.48, 74.51])
Sym = np.array([75.13, 77.04, 73.65, 73.88, 72.79, 70.29, 75.99, 73.18, 74.92, 66.99, 73.40, 75.01, 76.65, 73.29, 75.00, 73.13, 76.30, 74.43, 75.87, 74.16, 74.14, 75.92, 75.66, 73.26, 74.46, 74.96,
              75.66, 75.77, 74.94, 75.08])
Qym = np.array([74.59, 73.96, 75.26, 73.92, 76.17, 75.26, 76.82, 74.19, 68.07, 75.04, 77.24, 70.84, 75.20, 76.53, 71.51, 76.59, 73.45, 77.35, 73.15, 75.61, 74.98, 74.78, 76.60, 74.70, 74.23, 74.47,
              77.35, 74.03, 74.63, 76.15])
QiGi2=np.array([76.15, 72.96, 75.45, 73.63, 75.50, 75.18, 73.10, 72.49, 75.41, 74.11, 74.04, 75.08, 73.24, 76.63, 72.97, 71.24, 73.72, 76.47, 72.44, 74.89, 73.20, 72.27, 71.40, 73.02, 73.98, 74.56,
              73.65, 76.10, 73.36, 71.16])
DiGib=np.array([71.27, 74.75, 73.99, 74.88, 73.91, 73.32, 74.47, 73.58, 72.44, 74.16, 74.70, 72.54, 73.43, 71.87, 75.10, 75.57, 73.18, 74.08, 74.09, 72.56, 74.85, 73.93, 71.61, 72.02, 74.20, 73.70,
              73.70, 74.37, 74.32, 74.15])

models = [Scale, Sym, Qym, QiGi2, DiGib]
model_names = ['Scale', 'Sym', 'Qym', 'QiGi2', 'DiGib']

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