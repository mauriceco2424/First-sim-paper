import numpy as np
import matplotlib.pyplot as plt
import pyovf as OVF
from pathlib import Path
import skimage.filters
from scipy.signal import savgol_filter
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

def process_files(directory, min_x_wavelengths=6.5, time_start_ns=0, time_end_ns=np.inf):
    """Processes OVF files within a specified time interval, searching for max topological charge density with x constraints."""
    x_positions = []
    y_positions = []
    times = []
    overall_top_charges = []
    
    wavelength = 69.59e-9  # meters
    n_discre = 32
    a = wavelength / n_discre  # cell size in meters
    
    # Calculate the minimum x index based on the minimum x wavelength times the wavelength
    min_x_index = int(min_x_wavelengths * wavelength / a)

    file_paths = sorted(Path(directory).glob('*.ovf'), key=lambda path: int(path.stem[1:]))
    
    for file_path in file_paths:
        # Convert file index to time (in ns)
        current_time_ns = int(file_path.stem[1:]) * 1e-10 * 1e9
        
        # Filter based on the specified time interval
        if time_start_ns <= current_time_ns <= time_end_ns:
            data = read_ovf_data(file_path)
            vector_field = skimage.filters.gaussian(data[0], sigma=7, channel_axis=2)
            top_charge_density = calculate_topological_charge_density(vector_field)

            # Calculate the overall topological charge by summing the charge density
            overall_top_charge = np.sum(top_charge_density)
            overall_top_charges.append(overall_top_charge)
            
            # Limit the search area based on x constraints
            restricted_density = top_charge_density[:, min_x_index:]

            # Find the position of the maximum topological charge density in the restricted area
            max_index = np.unravel_index(np.argmax(restricted_density, axis=None), restricted_density.shape)
            y_positions.append(max_index[0] * a * 1e9)  # Convert to nanometers
            x_positions.append((max_index[1] + min_x_index) * a * 1e9)  # Adjust x position and convert to nanometers
            times.append(current_time_ns)

    return times, x_positions, y_positions, overall_top_charges

def smooth_data(data, window_length=11, polyorder=2):
    """Applies Savitzky-Golay filter to smooth the data."""
    return savgol_filter(data, window_length, polyorder)

def find_best_linear_fit(times, x_positions):
    """Finds the best linear fit for the smoothed position data."""
    smoothed_positions = smooth_data(x_positions)
    best_fit_slope = 0
    best_fit_intercept = 0
    best_fit_r_squared = -np.inf
    
    for window_size in range(10, len(times) // 2, 5):  # Vary window size
        for start in range(len(times) - window_size + 1):
            end = start + window_size
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(times[start:end], smoothed_positions[start:end])
            r_squared = r_value ** 2
            
            if r_squared > best_fit_r_squared:
                best_fit_slope = slope
                best_fit_intercept = intercept
                best_fit_r_squared = r_squared
    
    return best_fit_slope, best_fit_intercept, best_fit_r_squared

def calculate_velocity_and_acceleration(times, x_positions):
    """Calculates the velocity and acceleration based on the smoothed x positions over time using numerical derivatives."""
    smoothed_positions = smooth_data(x_positions)
    velocities = np.gradient(smoothed_positions, times)
    accelerations = np.gradient(velocities, times)
    return velocities, accelerations

def plot_positions(times, x_positions, output_directory, file_suffix, apply_linear_fit=False, best_fit_slope=None, best_fit_intercept=None):
    """Plots the x positions of maximum topological charge over time, optionally including the linear fit."""
    plt.figure(figsize=(18, 12))
    # Plot only markers, no connecting lines
    plt.scatter(times, x_positions, label=r'X Position', marker='o')  # Changed from plt.plot to plt.scatter

    if apply_linear_fit and best_fit_slope is not None and best_fit_intercept is not None:
        # Plot linear fit
        linear_fit = best_fit_slope * np.array(times) + best_fit_intercept
        plt.plot(times, linear_fit, label=r'Linear Fit', linestyle='--', color='red')

    # Set larger font sizes for labels, ticks, and title
    label_size = 30 
    tick_size = 30 

    plt.xlabel(r'Time [ns]', fontsize=label_size)
    plt.ylabel(r'Position [nm]', fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # Add shaded green areas for magnetic field pulses, but only label the first one
    pulse_times = [(60, 65), (85, 90), (110, 115)]
    plt.axvspan(pulse_times[0][0], pulse_times[0][1], color='green', alpha=0.3, label=r'70 mT Field Pulse')
    for start, end in pulse_times[1:]:
        plt.axvspan(start, end, color='green', alpha=0.3)

    # Insert horizontal orange dashed lines at 1330, 1610, and 1890 on the y-axis
    for y_value in [1330, 1610, 1890]:
        plt.axhline(y=y_value, color='orange', linestyle='--', linewidth=3, label=r'$4a_s$ Impurity' if y_value == 1330 else "")

    # Insert a red dashed horizontal line at 2205
    plt.axhline(y=2205, color='red', linestyle='-', linewidth=3, label=r'End of Track')

    # Create a custom legend
    plt.legend(fontsize=tick_size)

    plt.grid(True)
    plt.savefig(Path(output_directory) / f'positions_over_time_{file_suffix}.png', bbox_inches='tight')
    plt.close()



def plot_overall_topological_charge(times, overall_top_charges, output_directory, file_suffix):
    """Plots the overall topological charge over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(times, overall_top_charges, label='Overall Topological Charge', marker='o', color='blue')
    
    # Set larger font sizes for labels, ticks, and title
    label_size = 20 
    tick_size = 20 

    plt.xlabel('Time [ns]', fontsize=label_size)
    plt.ylabel('Topological Charge', fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    # Add vertical dashed lines for magnetic field pulses
    pulse_times = [(60, 65), (85, 90), (110, 115)]
    for start, end in pulse_times:
        plt.axvline(x=start, color='#006400', linestyle='--', linewidth=2)
        plt.axvline(x=end, color='#006400', linestyle='--', linewidth=2)

    plt.grid(True)
    plt.savefig(Path(output_directory) / f'overall_topological_charge_{file_suffix}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    directory = '/cluster/home/mauricec/simulations/6p5x6p5_25x4p5_res32_openBC_pin_0_4_8_4_run60ns_B001_n70mT_5ns_B0_20ns_B001_n70mT_5ns_B0_20ns_B001_n70mT_5ns_B0_20ns'
    output_directory = '/cluster/home/mauricec/simulations'
    file_suffix = 'pin_0_4_8_4_run60ns_B001_n70mT_5ns_B0_20ns_B001_n70mT_5ns_B0_20ns_B001_n70mT_5ns_B0_20ns'
    
    # Define the time interval you are interested in (e.g., 15ns to 25ns)
    time_start_ns = 0
    time_end_ns   = 135
    
    # Boolean flag to determine if linear fit should be applied to the position plot
    apply_linear_fit = False
    
    times, x_positions, y_positions, overall_top_charges = process_files(directory, time_start_ns=time_start_ns, time_end_ns=time_end_ns)
    
    best_fit_slope, best_fit_intercept, best_fit_r_squared = find_best_linear_fit(times, x_positions) if apply_linear_fit else (None, None, None)

    
    plot_positions(times, x_positions, output_directory, file_suffix, apply_linear_fit, best_fit_slope, best_fit_intercept)