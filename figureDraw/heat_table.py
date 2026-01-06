import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data from the table (mean values)
data = np.array([
    [78.3, 82.0, 82.2, 81.7, 76.2, 82.0, 80.5, 71.0, 71.1],  # CoraML base
    [76.8, 81.8, 82.1, 80.4, 77.1, 81.7, 82.1, 75.8, 75.5],  # CoraML ib
    [61.8, 66.0, 65.9, 65.8, 59.1, 66.6, 65.2, 56.2, 56.2],  # CiteSeer base
    [66.6, 66.0, 66.6, 66.5, 62.2, 66.1, 66.0, 59.1, 59.6],  # CiteSeer ib
    # Add your new dataset here (example):
    # [70.5, 75.0, 75.5, 74.8, 68.0, 76.0, 74.5, 65.0, 65.5],  # NewDataset base
    # [72.0, 76.5, 77.0, 75.5, 70.0, 77.5, 76.0, 67.0, 67.5],  # NewDataset ib
])

# Standard deviations
std = np.array([
    [1.4, 1.3, 0.8, 1.2, 2.9, 1.2, 1.4, 4.6, 4.7],  # CoraML base
    [2.1, 1.3, 1.6, 1.9, 2.3, 1.6, 1.5, 2.9, 3.0],  # CoraML ib
    [1.5, 1.7, 1.8, 2.2, 3.4, 2.0, 2.0, 2.8, 2.8],  # CiteSeer base
    [1.5, 2.0, 1.8, 2.0, 2.6, 1.7, 2.3, 2.7, 3.3],  # CiteSeer ib
    # Add your new dataset std here (example):
    # [1.8, 1.5, 1.2, 1.6, 2.5, 1.3, 1.7, 3.0, 3.2],  # NewDataset base
    # [1.9, 1.4, 1.3, 1.8, 2.2, 1.5, 1.6, 2.8, 3.1],  # NewDataset ib
])

# Row and column labels
row_labels = ['CoraML-base', 'CoraML-ib', 'CiteSeer-base', 'CiteSeer-ib',
              # Add your new dataset labels here (example):
              # 'NewDataset-base', 'NewDataset-ib'
             ]
col_labels = ['', 'dir', 'sym', 'row', '0', 'dir', 'sym', 'row', '0']

# Create figure
fig, ax = plt.subplots(figsize=(12, 5))

# Add title first
fig.suptitle('GNN Performance Heatmap (Row-wise Comparison)', fontsize=14, fontweight='bold', y=0.98)

# Create heatmap with row-wise normalization
# Normalize each row independently
data_normalized = np.zeros_like(data)
for i in range(data.shape[0]):
    row_min = data[i].min()
    row_max = data[i].max()
    if row_max > row_min:
        data_normalized[i] = (data[i] - row_min) / (row_max - row_min)
    else:
        data_normalized[i] = 0.5

# Create heatmap
im = ax.imshow(data_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks and labels
ax.set_xticks(np.arange(len(col_labels)))
ax.set_yticks(np.arange(len(row_labels)))
ax.set_xticklabels([])  # Hide the default x-axis labels
ax.set_yticklabels(row_labels)

# Add vertical lines to separate method groups
ax.axvline(x=0.5, color='black', linewidth=2)
ax.axvline(x=4.5, color='black', linewidth=2)

# Add text annotations with actual values and std
for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        text = ax.text(j, i, f'{data[i, j]:.1f}±{std[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=9, fontweight='bold')

# Add method group labels at top - aligned with their columns
ax.text(0, -0.85, 'DiG', ha='center', va='center', fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
ax.text(2.5, -0.85, '1iG', ha='center', va='center', fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
ax.text(7, -0.85, 'RiG', ha='center', va='center', fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

# Add column labels below model names
for j, label in enumerate(col_labels):
    if label:  # Only add non-empty labels
        ax.text(j, -0.35, label, ha='center', va='center', fontsize=9)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Performance (normalized per row)', rotation=270, labelpad=20)

# Title and layout
plt.tight_layout()

# Save the figure
plt.savefig('heatmap.pdf', dpi=300, bbox_inches='tight')
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
print("Heatmap saved as 'heatmap.pdf' and 'heatmap.png'")

plt.show()