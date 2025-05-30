import matplotlib.pyplot as plt

dataset = 'tel'

if dataset is 'cite':
    data_name = 'Citeseer'
    notself1_layers = list(range(1, 21))
    notself1_acc = [66.6, 63.1, 61.7, 61.4, 57.5, 51.6, 44.8, 28.9, 26.5, 21.9,
                    20.5, 21.6, 21.9, 20.6, 20.4, 20.7, 20.5, 19.5, 20.5, 19.9]
    notself1_std = [1.6, 2.2, 2.7, 3.2, 2.2, 5.4, 8.8, 7.0, 4.7, 4.8,
                    1.3, 1.4, 5.3, 1.7, 1.4, 1.4, 1.4, 1.8, 1.4, 1.6]

    addself_layers = list(range(1, 21))
    addself_acc = [65.3, 63.9, 61.1, 62.1, 57.8, 50.1, 41.6, 29.6, 26.8, 23.6,
                   22.5, 21.4, 22.0, 21.2, 21.5, 22.6, 20.5, 21.3, 21.6, 20.7]
    addself_std = [1.7, 2.6, 2.4, 2.9, 3.9, 11.9, 12.3, 5.6, 7.0, 5.4,
                   3.7, 2.9, 2.6, 1.6, 2.2, 4.0, 1.6, 1.5, 2.1, 2.2]

    rmself_layers = list(range(1, 21))
    rmself_acc = [64.4, 63.0, 61.3, 60.0, 56.1, 49.0, 40.5, 28.4, 21.8, 23.5,
                  22.5, 21.1, 21.2, 21.2, 20.3, 20.9, 20.8, 20.7, 19.8, 19.8]
    rmself_std = [2.1, 2.8, 2.6, 2.3, 6.8, 6.4, 4.6, 8.9, 3.6, 5.9,
                  5.3, 2.2, 1.6, 1.6, 2.1, 1.3, 1.1, 1.4, 1.8, 1.8]
elif dataset is 'cora':
    data_name = 'CoraML'

    notself1_layers = list(range(1, 21))
    notself1_acc = [72.1, 80.8, 80.8, 81.1, 78.9, 77.0, 76.0, 69.1, 67.2, 54.9,
                    45.6, 37.3, 35.7, 34.0, 32.3, 34.1, 33.6, 34.1, 34.7, 34.2]
    notself1_std = [3.6, 1.6, 1.4, 1.1, 2.1, 2.2, 3.5, 5.6, 3.9, 13.7,
                    15.9, 10.8, 3.9, 4.1, 4.6, 3.8, 3.8, 2.9, 4.5, 5.3]


    addself_layers = list(range(1, 21))
    addself_acc = [79.0, 82.1, 82.0, 80.8, 81.1, 77.4, 75.5, 70.3, 53.6, 48.7,
                   32.6, 33.5, 31.7, 33.4, 31.7, 33.6, 33.7, 33.4, 33.8, 32.3]
    addself_std = [1.5, 1.6, 1.1, 1.1, 1.5, 3.0, 3.3, 4.3, 18.2,
                   17.1, 2.1, 2.5, 2.3, 3.7, 2.5, 3.1, 3.2, 4.2, 3.4, 3.8]
elif dataset is 'wikics':
    data_name = 'WikiCS'

    notself1_layers = list(range(1, 21))
    notself1_acc = [66.4, 70.6, 70.4, 70.6, 70.6, 70.6, 70.6, 69.7, 69.7, 69.0,
                    67.0, 65.7, 62.0, 59.5, 58.5, 55.8, 47.5, 35.2, 29.7, 24.5]
    notself1_std = [0.8, 1.1, 1.2, 0.9, 1.2, 1.1, 1.4, 1.4, 1.5, 1.5,
                    1.9, 1.4, 5.7, 6.0, 8.8, 9.7, 12.0, 14.0, 9.8, 5.7]

    addself_layers = list(range(1, 21))
    addself_acc = [75.8, 76.6, 77.6, 77.5, 77.6, 76.5, 75.0, 73.5, 71.6, 69.7,
                   69.3, 66.6, 62.5, 61.0, 53.4, 55.9, 54.8, 52.0, 41.6, 32.7]
    addself_std = [0.4, 1.0, 0.6, 0.7, 0.8, 1.4, 1.7, 1.7, 1.9, 2.9,
                   2.1, 5.3, 6.5, 7.3, 7.1, 9.0, 11.3, 13.8, 17.2, 12.7]


elif dataset is 'pubmed':
    data_name = 'Pubmed'

    notself1_layers = list(range(1, 21))
    # notself1_acc = [61.5, 50.8, 46.1, 39.8, 42.7, 38.4, 38.6, 39.6, 42.7, 40.6,
    #                 40.8, 42.5, 36.3, 31.9, 33.5, 33.5, 35.0, 33.3, 28.6, 20.7]   # hid 10
    # notself1_std = [0.7, 5.3, 4.5, 8.6, 4.8, 3.2, 5.5, 3.4, 5.6, 5.2, 9.3,
    #                 4.9, 4.9, 5.3, 5.5, 5.1, 4.7, 4.9, 7.2, 0.0]


    addself_layers = list(range(1, 21))
    # addself_acc = [68.3, 59.8, 46.8, 42.2, 40.4, 42.1, 46.0, 43.0, 43.3, 41.2,
    #                47.3, 43.6, 34.9, 30.2, 34.2, 34.3, 34.0, 31.2, 27.1, 20.7]   # hid10
    # addself_std = [1.8, 4.2, 6.6, 7.9, 7.6, 8.0, 8.8, 8.4, 9.1, 9.1,
    #                7.8, 6.5, 11.4, 9.2, 11.2, 8.3, 7.5, 8.3, 7.9, 0.0]


elif dataset is 'tel':
    data_name = 'Telgram'

    notself1_layers = list(range(1, 21))
    notself1_acc = [50.8, 90.4, 92.4, 90.4, 89.8, 89.6, 87.0, 87.8, 86.6, 85.6,
                    86.6, 87.2, 85.6, 79.2, 84.8, 86.0, 81.2, 89.0, 85.8, 78.4]
    notself1_std = [8.8, 4.4, 3.6, 4.6, 5.5, 6.5, 8.1, 7.6, 5.8, 8.0,
                    3.9, 8.0, 7.2, 7.3, 9.4, 5.1, 6.1, 4.9, 6.6, 9.1]

elif dataset is 'chameleon':
    data_name = 'Chameleon'

    notself1_layers = list(range(1, 21))
    notself1_acc = [77.7, 78.1, 76.8, 75.8, 75.2, 74.2, 72.7, 71.8, 71.8, 70.2,
                    69.9, 69.5, 70.3, 69.5, 69.4, 69.8, 70.1, 69.8, 69.5, 69.4]
                    # 69.2, 69.9, 69.3]
    notself1_std = [1.0, 1.5, 1.3, 2.3, 2.4, 2.2, 2.5, 2.0, 2.2, 3.2,
                    2.1, 3.2, 2.2, 2.9, 3.5, 2.8, 2.8, 2.4, 2.6, 1.8]
                    # 3.7, 2.2, 3.3]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot Group 1 with error bars
plt.errorbar(notself1_layers, notself1_acc, yerr=notself1_std, fmt='o-',
             label='Not Selfloop', capsize=3, color='blue', markersize=4)

# Plot Group 2 with error bars
plt.errorbar(addself_layers, addself_acc, yerr=addself_std, fmt='o-',
             label='Add Selfloop', capsize=3, color='green', markersize=4)

# Plot Group 3 with error bars
plt.errorbar(rmself_layers, rmself_acc, yerr=rmself_std, fmt='o-',
             label='Remove Selfloop', capsize=3, color='red', markersize=4)

# Customize the plot
plt.xlabel('Number of Layers')
plt.ylabel('Accuracy (%)')
plt.title(data_name+' Accuracy vs Number of Layers')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Set y-axis limits to show full range of data
plt.ylim(0, 100)

# Add x-axis ticks for each layer
plt.xticks(notself1_layers)

# Show the plot
plt.tight_layout()
plt.savefig(data_name+"_accuracy_vs_layer.pdf", dpi=300)

plt.show()