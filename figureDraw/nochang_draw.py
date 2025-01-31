import numpy as np
import matplotlib.pyplot as plt
# data_name = 'Squirrel'
data_name = 'Chameleon'   # 'Squi'  revise here to draw

# Data
layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]
layers_squi = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50]

acc_mean_1_chame = [75.7, 76.8, 77.0, 75.3, 75.2, 74.7, 74.5, 74.3, 74.3, 74.1,
                    73.3, 73.1, 73.3, 73.7, 74.3, 72.8, 73.4, 73.1, 73.8, 73.9, 73.3, 74.2, 73.8]
acc_vari_1_chame = [1.3, 1.2, 1.4, 1.3, 2.3, 2.3, 2.1, 2.2, 2.2, 1.7,
                    2.1, 1.9, 2.5, 3.4, 2.6, 3.4, 2.4, 3.3, 2.5, 2.1, 2.4, 2.4, 2.9]

acc_mean_1_squi = [73.2, 72.4, 73.0, 72.9, 73.0, 73.0, 73.3, 73.0, 73.1, 73.1,
                  72.7, 72.7, 73.2, 73.5, 72.8, 72.9, 73.2, 73.5, 72.8, 72.9, 73.0, 72.3, 72.7]
acc_vari_1_squi = [1.8, 1.8, 2.0, 1.4, 1.9, 1.9, 1.9, 1.6, 1.7, 2.1,
                   1.8, 2.3, 1.8, 1.5, 1.6, 1.8, 1.6, 1.8, 2.0, 1.9, 1.4, 3.4, 1.6]

# Create figure and axis
plt.figure(figsize=(12, 6))

if data_name == 'Chameleon':
    # Plot Chameleon data
    plt.plot(layers, acc_mean_1_chame, 'b-', label='Chameleon', linewidth=2)
    plt.fill_between(layers,
                     np.array(acc_mean_1_chame) - np.array(acc_vari_1_chame),
                     np.array(acc_mean_1_chame) + np.array(acc_vari_1_chame),
                     color='blue', alpha=0.2)
elif data_name == 'Squirrel':
    # Plot Squirrel data
    plt.plot(layers_squi, acc_mean_1_squi, 'g-', label='Squirrel', linewidth=2)
    plt.fill_between(layers_squi,
                     np.array(acc_mean_1_squi) - np.array(acc_vari_1_squi),
                     np.array(acc_mean_1_squi) + np.array(acc_vari_1_squi),
                     color='green', alpha=0.2)


# Customize the plot
plt.xlabel('Number of Layers', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(data_name+ ' Accuracy by Number of Layers', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Set y-axis limits to better show the data
plt.ylim(45, 80)

# Add x-axis ticks
plt.xticks(np.concatenate([np.arange(1, 21, 2), np.array([30, 40, 50])]))

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.savefig(data_name+ " accuracy_multiple hop.pdf", dpi=300)
# Show the plot
plt.show()