import numpy as np
import matplotlib.pyplot as plt

data_name = 'Telegram'

layer_add_XW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000, 2000]
acc_mean_add_XW = [38.6,
    61.4, 91.2, 90.0, 92.0, 90.8, 91.0, 91.8, 89.6, 89.2, 89.0, 91.8, 90.8, 88.8, 90.2, 90.8, 89.0, 90.4, 89.6, 87.6, 88.0,
    88.6, 87.0, 89.6, 86.8, 86.2,
    86.8, 89.0, 86.6, 88.6, 91.0, 85.2, 85.6, 88.2, 87.8, 86.0, 88.8, 86.8, 81.2, 83.0, 86.4, 86.4, 84.6]
acc_std_add_XW = [6.7,
    8.1, 4.0, 2.3, 3.3, 5.9, 4.1, 4.7, 4.7, 5.9, 5.5, 2.2, 5.8, 4.0, 3.2, 4.7, 7.8, 4.5, 5.0, 4.5, 3.7,
6.1, 4.9, 3.6, 8.2, 6.4,
7.0, 4.1, 8.0, 5.7, 4.4, 6.0, 9.0, 5.9, 7.7, 4.3, 5.3, 4.1, 7.1, 7.6, 5.4, 5.4, 7.1]

layer_add_AX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000, 2000]
acc_mean_add_AX = [38.6,
    49.4, 91.2, 88.6, 90.0, 90.0, 92.2, 88.2, 89.8, 86.8, 89.6, 90.0, 89.8, 90.0, 89.4, 88.6, 88.4, 87.6, 88.6, 86.8, 88.2, 85.6, 84.0, 84.2, 80.8, 80.6, 80.6, 84.8, 80.4, 83.0, 80.0, 79.8, 81.4, 79.2, 76.8, 79.0, 82.2, 80.4, 80.6, 82.6, 83.8, 78.2, 81.2]
acc_std_add_AX = [6.7,
    7.5, 5.0, 5.1, 5.0, 3.9, 3.7, 6.1, 5.3, 4.3, 2.8, 5.2, 3.9, 5.6, 6.8, 4.4, 7.1, 3.9, 6.8, 5.0, 5.9, 8.6, 6.9, 7.1, 9.9, 9.3, 11.7, 6.7, 10.1, 10.8, 8.1, 10.6, 9.9, 9.2, 11.0, 10.6, 10.5, 10.9, 7.7, 8.6, 6.8, 10.5, 8.0]

layer_noadd_AX_Jan = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400]
acc_mean_noadd_AX_Jan = [38.6,
    50.8, 90.8, 92.6, 91.2, 90.6, 90.0, 87.0, 91.2, 88.0, 90.0, 90.0, 86.6, 89.2, 87.8, 87.2, 88.0, 89.2, 88.0, 87.6, 89.8,
83.8, 84.2, 88.4, 85.8, 80.6, 84.2, 80.0, 84.2, 85.0, 84.6, 79.4, 84.4, 82.8, 86.4, 87.2, 81.6, 80.0, 84.2
                  ]
acc_std_noadd_AX_Jan = [6.7,
    8.8, 5.6, 4.2, 6.2, 5.4, 3.4, 6.6, 2.3, 6.7, 3.4, 3.3, 6.5, 3.0, 7.4, 6.8, 4.7, 3.0, 5.3, 4.9, 3.9,
                  8.4, 7.9, 5.9, 7.9, 6.0, 10.5, 7.0, 6.5, 8.7, 6.5, 7.2, 6.9, 6.6, 6.6, 8.1, 8.8, 7.0, 10.5
                  ]


layer_noadd_AX_430 = [0,
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000, 2000]
acc_mean_noadd_AX_430 = [38.6,
50.8, 92.2, 92.2, 90.2, 90.2, 91.8, 90.4, 89.0, 89.6, 90.0, 87.6, 90.6, 86.6, 87.6, 87.8, 86.8, 89.0, 87.4, 88.0, 88.4, 83.2, 83.6, 85.0, 84.8, 79.2, 86.2, 80.4, 83.0, 84.6, 85.0, 79.2, 84.0, 81.6, 84.4, 87.2, 81.6, 81.0, 87.2, 78.6, 82.4, 81.4, 82.8]
acc_std_noadd_AX_430 = [6.7,
8.8, 3.8, 3.8, 6.8, 6.0, 4.4, 4.2, 5.1, 4.2, 4.3, 5.7, 5.3, 8.2, 6.9, 6.3, 5.8, 6.4, 6.0, 6.5, 2.8, 7.2, 7.9, 7.8, 7.6, 8.5, 8.1, 9.6, 9.2, 8.5, 9.0, 6.8, 6.3, 6.9, 7.0, 8.1, 8.8, 7.3, 7.4, 7.2, 7.8, 10.9, 6.6]

layer_noadd_XW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000, 2000]
acc_mean_noadd_XW = [38.6,
    63.6, 91.4, 90.2, 91.2, 89.4, 92.0, 91.4, 92.4, 90.4, 89.8, 88.2, 90.6, 87.4, 89.8, 90.0, 88.4, 89.4, 89.2, 87.2, 87.8, 88.0, 88.6, 91.6, 90.0, 89.2, 89.0, 89.2, 89.6, 88.6, 90.8, 91.4, 88.8, 91.2, 89.2, 88.8, 90.6, 90.4, 87.8, 89.8, 87.8, 88.8, 90.8]
acc_std_noadd_XW = [6.7,
    6.6, 4.3, 3.9, 3.6, 4.7, 4.3, 4.0, 3.9, 4.4, 2.9, 4.8, 4.6, 7.9, 4.3, 5.6, 5.8, 6.0, 5.9, 3.3, 5.5, 5.5, 6.4, 3.7, 5.6, 5.0, 4.5, 7.6, 5.0, 4.9, 4.7, 2.7, 4.3, 3.2, 6.7, 5.6, 3.1, 3.9, 4.3, 7.7, 5.5, 3.2, 4.8]

plt.figure(figsize=(12, 6))

# Improve x-axis ticks - fewer, more spread out ticks
# combined_ticks = sorted(set(layer_add_AX + layer_add_XW + layer_noadd_AX_Jan +
#                             # layer_noadd_AX_430 +
#                             layer_noadd_XW))
# plt.xticks(combined_ticks)
plt.xticks([0, 10, 25, 50, 75, 100, 150, 200, 300, 400])
# plt.xticks(np.array([0, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, ]))

plt.plot(layer_add_XW, acc_mean_add_XW, 'o-', color='blue',label='addSelfloop+XW', linewidth=2)
plt.fill_between(layer_add_XW,
                 np.array(acc_mean_add_XW) - np.array(acc_std_add_XW),
                 np.array(acc_mean_add_XW) + np.array(acc_std_add_XW),
                 color='blue', alpha=0.2)

plt.plot(layer_add_AX, acc_mean_add_AX, 'o-', color='black', label='addSelfloop+AX', linewidth=2)
plt.fill_between(layer_add_AX,
                 np.array(acc_mean_add_AX) - np.array(acc_std_add_AX),
                 np.array(acc_mean_add_AX) + np.array(acc_std_add_AX),
                 color='black', alpha=0.2)

plt.plot(layer_noadd_XW, acc_mean_noadd_XW, '-', color='blue',label='noaddSelfloop+XW', linewidth=2)
plt.fill_between(layer_noadd_XW,
                 np.array(acc_mean_noadd_XW) - np.array(acc_std_noadd_XW),
                 np.array(acc_mean_noadd_XW) + np.array(acc_std_noadd_XW),
                 color='blue', alpha=0.2)

plt.plot(layer_noadd_AX_430, acc_mean_noadd_AX_430, '-',color='black', label='noaddSelfloop+AX', linewidth=2)
plt.fill_between(layer_noadd_AX_430,
                 np.array(acc_mean_noadd_AX_430) - np.array(acc_std_noadd_AX_430),
                 np.array(acc_mean_noadd_AX_430) + np.array(acc_std_noadd_AX_430),
                 color='black', alpha=0.2)

# plt.plot(layer_noadd_AX_Jan, acc_mean_noadd_AX_Jan, 'g-', label='noaddSelfloop+AX', linewidth=2)
# plt.fill_between(layer_noadd_AX_Jan,
#                  np.array(acc_mean_noadd_AX_Jan) - np.array(acc_std_noadd_AX_Jan),
#                  np.array(acc_mean_noadd_AX_Jan) + np.array(acc_std_noadd_AX_Jan),
#                  color='green', alpha=0.2)



# Customize the plot
plt.xlabel('Number of Layers', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title(data_name + ' Accuracy by Number of Layers', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='lower right')
# plt.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)

plt.xticks(rotation=5)

plt.ylim(30, 100)
plt.xlim(0, 50)




plt.tight_layout()
plt.savefig(data_name+ "2- accuracy_multiple hop100.pdf", dpi=300)
plt.show()