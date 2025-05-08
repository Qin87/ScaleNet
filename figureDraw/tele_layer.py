import numpy as np
import matplotlib.pyplot as plt

addSloop = 1
data_name = 'telegram'   # 'Squi'  revise here to draw

if addSloop == 0:
    # rawfiles in telegram_Direct_BNorm_ScaleNet64hid__Bal_dir-gcn_part0.5_-1_-1_sloop00_jk0_norm0_lay1_lr0.01_NoImp410q0_20250108141953.log
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400]

    acc_mean_noadd = [38.6,
        50.8, 90.8, 92.6, 91.2, 90.6, 90.0, 87.0, 91.2, 88.0, 90.0, 90.0, 86.6, 89.2, 87.8, 87.2, 88.0, 89.2, 88.0, 87.6, 89.8,
        83.8, 84.2, 88.4, 85.8, 80.6, 84.2, 80.0, 84.2, 85.0, 84.6, 79.4, 84.4, 82.8, 86.4, 87.2, 81.6, 80.0, 84.2
    ]
    acc_vari_noadd = [6.7,
        8.8, 5.6, 4.2, 6.2, 5.4, 3.4, 6.6, 2.3, 6.7, 3.4, 3.3, 6.5, 3.0, 7.4, 6.8, 4.7, 3.0, 5.3, 4.9, 3.9,
        8.4, 7.9, 5.9, 7.9, 6.0, 10.5, 7.0, 6.5, 8.7, 6.5, 7.2, 6.9, 6.6, 6.6, 8.1, 8.8, 7.0, 10.5
    ]
elif addSloop == 1:
    layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 100, 200, 300, 400, 500, 600, 1000, 2000]
    acc_mean_add = [38.6, 61.4, 91.2, 90.0, 92.0, 90.8, 91.0, 91.8, 89.6, 89.2, 89.0, 91.8, 90.8, 88.8, 90.2, 90.8, 89.0, 90.4, 89.6, 87.6, 88.0, 87.8, 88.8, 86.8, 80.6, 83.0, 86.4, 87.4, 86.0]
    acc_vari_add = [6.7, 8.1, 4.0, 2.3, 3.3, 5.9, 4.1, 4.7, 4.7, 5.9, 5.5, 2.2, 5.8, 4.0, 3.2, 4.7, 7.8, 4.5, 5.0, 4.5, 3.7, 7.2, 5.3, 4.1, 9.1, 7.5, 5.4, 7.1, 7.4]

# elif data_name == 'telegram':
#     # Plot Telegram data
plt.plot(layers, acc_mean_noadd, 'r-', label='Telegram', linewidth=2)
plt.fill_between(layers,
                 np.array(acc_mean_noadd) - np.array(acc_vari_noadd),
                 np.array(acc_mean_noadd) + np.array(acc_vari_noadd),
                 color='red', alpha=0.2)


plt.xlabel('Number of Layers', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(data_name + ' Accuracy by Number of Layers', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right', fontsize=10)  # Position legend in upper right

# Set y-axis limits to better show the data
plt.ylim(30, 100)
plt.xlim(0, 400)

# Improve x-axis ticks - fewer, more spread out ticks
plt.xticks(np.array([0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, ]))

# Rotate x-axis labels less for better readability
plt.xticks(rotation=5)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.savefig(data_name + "_accuracy_by_layers.pdf", dpi=300)
# Show the plot
plt.show()