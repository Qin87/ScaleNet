import matplotlib.pyplot as plt
import numpy as np

# Data
times = [200, 160, 150, 140, 120, 110, 100, 50, 40, 32, 20, 16, 8, 4, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.01, 0.005, 0.001, 0.0005, 0.00005, 5e-6, 3e-6, 2e-6, 1e-6, 9e-7, 5e-7, 5e-8]
accuracy = [26.5, 27.2, 30.7, 40.8, 39, 25, 30, 36.2, 41.5, 39.7, 49.3, 44.1, 42.3, 50.2, 44.5, 32.9, 51.8, 47.8, 31.8, 65.6, 65.4, 68.6, 65.1, 63.2, 51.3, 51.1, 44.7, 36.6, 36.4, 36.8, 36.8, 32.2, 22.4, 22.4, 22.4, 22.4]

# Create figure and axis
plt.figure(figsize=(12, 6))

# Create the plot with times on logarithmic x-axis
plt.semilogx(times, accuracy, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)

# Add labels and title
plt.xlabel('Coefficient (log scale)', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Accuracy vs. Coefficient (Logarithmic Scale)', fontsize=14)

# Add grid
plt.grid(True, which="both", ls="-", alpha=0.2)

# Set x-axis limits to make sure all points are visible
plt.xlim(max(times) * 1.1, min(times) * 0.9)

# Add annotations for peak accuracy
max_accuracy = max(accuracy)
max_index = accuracy.index(max_accuracy)
max_time = times[max_index]
plt.annotate(f'Max: {max_accuracy}% at t={max_time}',
             xy=(max_time, max_accuracy),
             xytext=(max_time*2, max_accuracy + 5),
             arrowprops=dict(arrowstyle='->'),
             fontsize=10)

# Improve appearance
plt.tight_layout()

# Show the plot
plt.show()