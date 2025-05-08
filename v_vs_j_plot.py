import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'velocity_vs_current_5x5_7x7.xlsx'
df = pd.read_excel(file_path)  # Assuming the data starts from the first row

# Rename columns for ease of use
df.columns = ['Current (10^10 A/m^2)', 
              'v in m/s pol=0.2, 5x5_20x3', 'v in m/s pol=0.4, 5x5_20x3', 'v in m/s pol=0.6, 5x5_20x3', 
              'v in m/s pol=0.8, 5x5_20x3', 'v in m/s pol=1.0, 5x5_20x3',
              'v in m/s pol=0.2, 7x7_20x5']

# Create a plot
plt.figure(figsize=(18, 18))

# Specify which columns to plot
columns_to_plot = [
    'v in m/s pol=0.2, 5x5_20x3', 
    'v in m/s pol=0.4, 5x5_20x3', 
    'v in m/s pol=0.2, 7x7_20x5'
]

# Extract and plot data for each specified polarization
for velocity_col in columns_to_plot:
    # Extract the geometry and polarization values for the legend
    pol_label = velocity_col.split(", ")[1]  # Get geometry part directly (e.g., '5x5_20x3' or '7x7_20x5')
    pol_value = velocity_col.split(", ")[0].replace('v in m/s pol=', 'P=')  # Change 'pol=' to 'P=' for polarization part
    new_label = f'{pol_label}, {pol_value}'  # Combine geometry part with polarization part for legend
    plt.plot(df['Current (10^10 A/m^2)'], df[velocity_col], label=new_label, marker='o', markersize=40)

label_size  = 60
tick_size   = 60
legend_size = 50

plt.xlabel(r'$j_y$ [$10^{10}$ A/m$^2$]', fontsize=label_size)
plt.ylabel(r'$v_x$ [m/s]', fontsize=label_size)

# Update tick parameters
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)

# Update legend
plt.legend(fontsize=legend_size)

# Display the grid
plt.grid(True)

# Save the plot as a PNG file
plt.savefig('velocity_vs_current_plot.png', bbox_inches='tight')
