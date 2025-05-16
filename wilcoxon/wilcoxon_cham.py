import numpy as np
from scipy.stats import wilcoxon

# These arrays should contain the paired data
Scale111 = np.array([77.19, 81.58, 79.17, 79.61, 78.07, 80.26, 82.68, 78.73, 78.73, 78.95, 77.19, 80.48, 77.41, 78.51, 78.95, 78.95, 80.92, 77.85, 79.17, 80.70, 78.07, 81.14, 78.07, 76.75, 76.32, 80.48, 82.24, 78.95, 81.14, 79.82])
Scale1_11 = np.array([76.54, 80.92, 76.54, 76.75, 77.63, 78.07, 78.51, 78.51, 79.61, 77.85, 74.56, 82.02, 76.75, 75.00, 77.85, 78.29, 80.48, 76.97, 77.85, 78.51, 76.54, 80.04, 75.88, 77.41, 77.41, 79.82, 79.82, 77.85, 78.95, 77.85])
Scale1_1_1 = np.array([78.07, 80.70, 77.41, 78.07, 78.29, 80.48, 78.51, 78.29, 77.85, 78.07, 77.19, 81.14, 79.17, 78.51, 77.19, 79.61, 80.26, 79.17, 77.85, 78.95, 75.44, 81.80, 78.73, 77.85, 76.54, 79.17, 78.51, 78.29, 79.17, 77.41])
DirGNN = np.array([74.56, 80.26, 76.97, 76.97, 78.51, 80.04, 78.73, 77.19, 78.73, 78.51, 77.85, 79.61, 79.82, 77.41, 77.19, 77.63, 77.63, 78.73, 78.29, 79.61, 76.97, 80.48, 78.73, 78.51, 77.19, 79.39, 78.29, 80.48, 77.63, 79.39])
QiG = np.array([70.39, 73.68, 70.18, 69.96, 73.46, 72.37, 69.08, 68.86, 74.12, 71.05, 69.74, 73.03, 69.30, 69.30, 73.25, 71.27, 67.76, 67.32, 74.12, 71.93, 70.83, 72.81, 71.93, 70.18, 72.15, 71.71, 68.86, 68.42, 73.25, 71.27])


models = [Scale111, Scale1_11, Scale1_1_1, DirGNN, QiG]
model_names = ['Scale111', 'Scale1_11', 'Scale1_1_1', 'DirGNN', 'QiG']

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