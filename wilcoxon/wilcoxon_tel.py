import numpy as np
from scipy.stats import wilcoxon

# These arrays should contain the paired data
Scale=np.array([98, 94.00, 98.00, 94.00, 98.00, 100.00, 94.00, 96.00, 98.00, 98.00, 90.00, 94.00, 96.00, 100.00, 98.00, 92.00, 96.00, 96.00, 94.00, 94.00, 98.00, 90.00, 100.00, 100.00, 100.00, 90.00, 94.00, 94.00, 96.00, 98.00])
QiG=np.array([100.00, 96.00, 92.00, 92.00, 98.00, 100.00, 90.00, 96.00, 96.00, 92.00, 90.00, 94.00, 94.00, 98.00, 88.00, 96.00, 82.00, 90.00, 96.00, 90.00, 92.00, 94.00, 98.00, 88.00, 96.00, 98.00, 84.00, 90.00, 94.00, 94.00])
DirGNN=np.array([88.00, 90.00, 94.00, 96.00, 98.00, 86.00, 94.00, 98.00, 94.00, 84.00, 92.00, 94.00, 92.00, 96.00, 96.00, 90.00, 92.00, 92.00, 96.00, 84.00, 94.00, 88.00, 92.00, 90.00, 96.00, 90.00, 92.00, 90.00, 96.00, 92.00])
Mag=np.array([86.00, 76.00, 96.00, 92.00, 86.00, 78.00, 90.00, 84.00, 92.00, 78.00, 88.00, 80.00, 94.00, 90.00, 90.00, 84.00, 90.00, 84.00, 86.00, 84.00, 90.00, 84.00, 96.00, 92.00, 90.00, 82.00, 86.00, 84.00, 90.00, 76.00 ])
# Qym=np.array([])

models = [Scale,QiG, DirGNN, Mag]
model_names = ['Scale', 'QiG', 'DirGNN', 'Mag']

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