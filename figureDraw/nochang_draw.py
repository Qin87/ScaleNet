import numpy as np
import matplotlib.pyplot as plt
# data_name = 'Squirrel'
# data_name = 'Chameleon'   # 'Squi'  revise here to draw
data_name = 'Telegram'   # revise here to draw

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

layers_tel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400]
acc_mean_1_tel = [38.6,
    50.8, 90.8, 92.6, 91.2, 90.6, 90.0, 87.0, 91.2, 88.0, 90.0, 90.0, 86.6, 89.2, 87.8, 87.2, 88.0, 89.2, 88.0, 87.6, 89.8,
    83.8, 84.2, 88.4, 85.8, 80.6, 84.2, 80.0, 84.2, 85.0, 84.6, 79.4, 84.4, 82.8, 86.4, 87.2, 81.6, 80.0, 84.2
]
acc_vari_1_tel = [6.7,
    8.8, 5.6, 4.2, 6.2, 5.4, 3.4, 6.6, 2.3, 6.7, 3.4, 3.3, 6.5, 3.0, 7.4, 6.8, 4.7, 3.0, 5.3, 4.9, 3.9,
    8.4, 7.9, 5.9, 7.9, 6.0, 10.5, 7.0, 6.5, 8.7, 6.5, 7.2, 6.9, 6.6, 6.6, 8.1, 8.8, 7.0, 10.5
]

# Create figure and axis
plt.figure(figsize=(12, 6))

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
elif data_name == 'Telegram':
    # Customize the plot
    plt.xlabel('Number of Layers', fontsize=6)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(data_name + ' Accuracy by Number of Layers', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.xticks(rotation=5)

    plt.ylim(30, 100)
    plt.xlim(0, 400)

    # Improve x-axis ticks - fewer, more spread out ticks
    plt.xticks(np.array([0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, ]))

    # Rotate x-axis labels less for better readability
    plt.xticks(rotation=5)

    plt.plot(layers_tel, acc_mean_1_tel, 'r-', label='Telegram', linewidth=2)
    plt.fill_between(layers_tel,
                     np.array(acc_mean_1_tel) - np.array(acc_vari_1_tel),
                     np.array(acc_mean_1_tel) + np.array(acc_vari_1_tel),
                     color='red', alpha=0.2)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.savefig(data_name+ " accuracy_multiple hop.pdf", dpi=300)
# Show the plot
plt.show()