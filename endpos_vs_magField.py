import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '/cluster/home/mauricec/simulations/end_position_vs_magnField.xlsx'
df = pd.read_excel(file_path)

# Increase the size of labels, ticks, and general font by 30%
plt.rcParams.update({
    'axes.labelsize':  30,  # Axis label size increased by 30%
    'axes.titlesize':  30,  # Title size increased by 30%
    'xtick.labelsize': 30,  # X tick label size increased by 30%
    'ytick.labelsize': 30,  # Y tick label size increased by 30%
    'legend.fontsize': 30  # Legend font size increased by 30%
})

# Create a plot with increased size
plt.figure(figsize=(18, 10))  # Adjust the size as needed

# Correct column names based on your error message
magnetic_field_col = 'Magnetic field (mT)'  # Updated based on your error message
endpos_col = 'End position (nm)'  # Assuming the velocity column name is correct

# Plot the data with larger markers (doubling the size)
plt.plot(df[magnetic_field_col], df[endpos_col], marker='o', linestyle='-', markersize=16)

# Add horizontal lines with updated labels using LaTeX for lambda/16, lambda/8, and lambda/4
# Increased line thickness (linewidth) to make dashed lines thicker
plt.axhline(y=1050, color='g', linestyle='--', linewidth=3, label=r'$2a_s$ Impurity')
plt.axhline(y=1330, color='c', linestyle='--', linewidth=3, label=r'$4a_s$ Impurity')
plt.axhline(y=1610, color='orange', linestyle='--', linewidth=3, label=r'$6a_s$ Impurity')
plt.axhline(y=1750, color='red', linestyle='-', linewidth=3, label='End of Track')

plt.legend(loc='lower right')

# Labeling the axes with LaTeX for subscripts and increased size
plt.xlabel(r'$-B_z$ [mT]')  # X axis label size increased by 30%
plt.ylabel(r'End position $x_{max}$ [nm]')  # Y axis label size increased by 30%

plt.ylim(bottom=None, top=1800)  # Adjust the top value to go higher than 1750

# Adding grid
plt.grid(True)

# Save the plot to a file with increased resolution
plt.savefig('/cluster/home/mauricec/simulations/endpos_vs_mfield_plot.png', dpi=300, bbox_inches='tight')
