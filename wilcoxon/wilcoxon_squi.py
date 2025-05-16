import numpy as np
from scipy.stats import wilcoxon

# These arrays should contain the paired data
Scale111 = np.array([75.22, 76.85, 74.83, 77.91, 75.89, 75.12, 73.39, 75.79, 70.70, 75.98, 75.31, 76.95, 75.12, 78.19, 76.56, 74.64, 74.06, 75.79, 71.47, 77.52, 74.16, 76.37, 74.83, 77.71, 74.74, 75.22, 73.68, 75.70, 71.57, 77.91])
Scale11_1 = np.array([74.35, 75.41, 73.68, 77.04, 73.68, 74.83, 72.14, 75.89, 71.37, 75.98, 74.64, 77.33, 74.93, 77.71, 73.58, 74.45, 71.95, 75.89, 70.41, 78.10, 72.53, 74.93, 75.02, 76.95, 74.64, 76.75, 74.83, 73.87, 73.87, 75.70])
Scale1_1_1 =np.array([74.74, 75.50, 74.93, 77.14, 74.35, 74.26, 74.54, 74.54, 69.36, 77.43, 74.74, 76.56, 73.87, 76.08, 75.02, 74.83, 73.49, 75.60, 74.54, 75.98, 73.29, 77.33, 73.97, 75.02, 73.20,
                      75.60, 69.74, 77.52, 69.74, 77.52])
DirGNN = np.array([72.62, 75.22, 74.35, 77.04, 75.02, 75.60, 73.49, 75.89, 71.95, 77.62, 74.26, 75.50, 72.62, 77.81, 72.91, 76.27, 72.81, 75.02, 71.66, 77.43, 74.64, 76.08, 74.35, 78.19, 76.46, 76.27, 73.39, 75.60, 71.95, 78.00])

models = [Scale111, Scale11_1, Scale1_1_1, DirGNN]
model_names = ['Scale111', 'Scale11_1', 'Scale1_1_1', 'DirGNN']

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