import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'Analyse_Geometries.xlsx'
df = pd.read_excel(file_path)

# Prepare the data by alternating between 'Width' and 'Result'
sizes = df.columns[1::2]  # Size columns
results = df.columns[2::2]  # Result columns

# Extract numerical values for square sizes from the column names
size_values = [float(size.split('x')[0]) for size in sizes]

# Define the improved color map
color_map = {'Y': '#2196F3', 'N': '#F44336', 'R': '#4CAF50'}
legend_labels = {'Y': "Centered, stable", 'N': "Off-centered", 'R': "Centered, relaxing"}
max_width = 9.5
min_width = 2.5
width_values = np.arange(min_width, max_width + 0.5, 0.5)

# Create a dataframe to hold the color data
color_grid = pd.DataFrame(index=width_values, columns=size_values, dtype='object')

# Populate the color grid
for size, result in zip(sizes, results):
    size_value = float(size.split('x')[0])
    for _, row in df[[size, result]].dropna().iterrows():
        if row[size] in width_values:
            color = color_map.get(row[result], None)
            if color:
                color_grid.at[row[size], size_value] = color

# Plotting the color grid with swapped axes
fig, ax = plt.subplots(figsize=(18, 16))
gap = 0.05  # Set a small gap between rectangles
for i, width in enumerate(color_grid.index):
    for j, size in enumerate(color_grid.columns):
        color = color_grid.at[width, size]
        if pd.notna(color):
            rect = plt.Rectangle((j + gap/2, i + gap/2), 1 - gap, 1 - gap, color=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

# Create patches for the legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_map['Y'], edgecolor='black', label=legend_labels['Y']),
                   Patch(facecolor=color_map['R'], edgecolor='black', label=legend_labels['R']),
                   Patch(facecolor=color_map['N'], edgecolor='black', label=legend_labels['N'])]

# Adding the legend to the plot
ax.legend(handles=legend_elements, loc='upper left', fontsize=30)

# Set x and y ticks and labels
ax.set_xticks(np.arange(len(size_values)) + 0.5)
ax.set_yticks(np.arange(len(width_values)) + 0.5)
ax.set_xticklabels(size_values, fontsize=30)
ax.set_yticklabels(width_values, fontsize=30)

# Adding axis labels
ax.set_xlabel(r'Square side length $d_1$ [$\lambda$]', fontsize=30)  # Increase fontsize for axis label
ax.set_ylabel(r'Width of track $d_3$ [$\lambda$]', fontsize=30)  # Increase fontsize for axis label

ax.set_xlim(0, len(size_values))
ax.set_ylim(0, len(width_values))

ax.grid(False)
plt.savefig('analyse_geometry.png', bbox_inches='tight')

