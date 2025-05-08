import numpy as np
import matplotlib.pyplot as plt
import pyovf as OVF
from pathlib import Path
import skimage.filters
import scipy.stats

def read_ovf_data(file_path):
    """Reads an OVF file and extracts the magnetic vector data."""
    _, _, data = OVF.read(str(file_path))
    return data

def calculate_topological_charge_density(vector_field):
    """Calculates the topological charge density of the vector field."""
    dx = np.gradient(vector_field, axis=0)
    dy = np.gradient(vector_field, axis=1)
    cross_product = np.cross(dx, dy)
    top_charge_density = np.sum(vector_field * cross_product, axis=2) / (4 * np.pi)
    return top_charge_density

def process_files(directory, min_x_wavelengths=5):
    """Processes all OVF files in the directory, searching for max topological charge density with x constraints."""
    x_positions = []
    y_positions = []
    times = []
    
    wavelength = 69.59e-9  # meters
    n_discre = 32
    a = wavelength / n_discre  # cell size in meters
    
    min_x_index = int(min_x_wavelengths * wavelength / a)

    file_paths = sorted(Path(directory).glob('*.ovf'), key=lambda path: int(path.stem[1:]))
    for file_path in file_paths:
        data = read_ovf_data(file_path)
        vector_field = skimage.filters.gaussian(data[0], sigma=7, channel_axis=2)
        top_charge_density = calculate_topological_charge_density(vector_field)

        restricted_density = top_charge_density[:, min_x_index:]
        max_index = np.unravel_index(np.argmax(restricted_density, axis=None), restricted_density.shape)
        y_positions.append(max_index[0] * a * 1e9)
        x_positions.append((max_index[1] + min_x_index) * a * 1e9)
        times.append(int(file_path.stem[1:]) * 1e-10 * 1e9)

    return times, x_positions, y_positions

def plot_positions(times, x_positions, output_directory, file_suffix):
    """Plots the x positions of maximum topological charge over time, including a threshold line."""
    plt.figure(figsize=(12, 6))
    plt.plot(times, x_positions, marker='.', linestyle='None')
    
    # Plot threshold lines at specific y positions
    plt.axhline(y=1050, color='g', linestyle='--', label='Position of first impurity')
    plt.axhline(y=1330, color='c', linestyle='--', label='Position of second impurity')
    plt.axhline(y=1610, color='orange', linestyle='--', label='Position of third impurity')
    plt.axhline(y=1750, color='red', linestyle='--', label='End of track')
    
    plt.ylim(bottom=None, top=1800)  # Adjust the top value to go higher than 1750

    plt.xlabel('Time (ns)')
    plt.ylabel('Position (nm)')
    plt.title('Position of Maximum Topological Charge Over Time')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(Path(output_directory) / f'positions_over_time_{file_suffix}.png')
    plt.close()

def check_pinned(x_positions):
    """Checks if the final position is greater than 750 nm."""
    return "pinned: true" if x_positions[-1] < 1610 else "pinned: false"

if __name__ == "__main__":
    directory = '/cluster/home/mauricec/simulations/pin_0_2cs_4_4cs_6_8cs_Bz_n70mT_5ns_B0_5ns_Bz_n80mT_40ns'
    output_directory = '/cluster/home/mauricec/simulations'
    file_suffix = 'pin_0_2cs_4_4cs_6_8cs_Bz_n70mT_5ns_B0_5ns_Bz_n80mT_40ns'
    
    times, x_positions, y_positions = process_files(directory)
    pinned_status = check_pinned(x_positions)
    print(pinned_status)
    print(f"Last X Position: {x_positions[-1]} nm")
    
    plot_positions(times, x_positions, output_directory, file_suffix)
