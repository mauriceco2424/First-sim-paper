import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'velocity_vs_mfield_5x5_7x7.xlsx'
df = pd.read_excel(file_path)  # Assuming no skiprows are needed

# Rename columns correctly based on your Excel sheet
df.columns = ['Magnetic Field (mT)', 
              'v in m/s (7x7)', 'v in m/s (5x5)']

# Create a plot
plt.figure(figsize=(18, 18))

# Define custom labels for the datasets
labels = {
    'v in m/s (7x7)': '7x7_20x5',
    'v in m/s (5x5)': '5x5_20x3'
}

# Extract and plot data for each dataset
for velocity_col in df.columns[1:]:  # Exclude the first column which is 'Magnetic Field'
    plt.plot(df['Magnetic Field (mT)'], df[velocity_col], label=labels[velocity_col], marker='o', markersize=40)

label_size  = 60
tick_size   = 60
legend_size = 50

plt.xlabel(r'$-B_z$ [mT]', fontsize=label_size)
plt.ylabel(r'$v_x$ [m/s]', fontsize=label_size)

# Update tick parameters
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

# Update legend
plt.legend(fontsize=legend_size)

# Display the grid
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('velocity_vs_mfield_plot.png', bbox_inches='tight')
