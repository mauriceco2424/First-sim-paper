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

def process_files(directory, min_x_wavelengths=5.5, time_start_ns=0, time_end_ns=np.inf):
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

def find_two_linear_fits(times, x_positions):
    """Finds the best two linear regions in the data and fits a line to both."""
    smoothed_positions = smooth_data(x_positions)
    
    best_split_index = None
    best_r_squared_total = -np.inf
    best_fit_slopes = [0, 0]
    best_fit_intercepts = [0, 0]
    
    # Loop over possible split points (avoid splitting too close to the ends)
    for split_index in range(10, len(times) - 10):
        # First region
        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = scipy.stats.linregress(
            times[:split_index], smoothed_positions[:split_index]
        )
        r_squared_1 = r_value_1 ** 2
        
        # Second region
        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = scipy.stats.linregress(
            times[split_index:], smoothed_positions[split_index:]
        )
        r_squared_2 = r_value_2 ** 2
        
        # Combine the R^2 values (you could use a weighted sum if needed)
        r_squared_total = r_squared_1 + r_squared_2
        
        # Update the best split if this split is better
        if r_squared_total > best_r_squared_total:
            best_split_index = split_index
            best_r_squared_total = r_squared_total
            best_fit_slopes = [slope_1, slope_2]
            best_fit_intercepts = [intercept_1, intercept_2]
    
    return best_split_index, best_fit_slopes, best_fit_intercepts, best_r_squared_total

def plot_positions(times, x_positions, y_positions, output_directory, file_suffix, apply_linear_fit=False, best_fit_slope=None, best_fit_intercept=None, start_time_for_fit=20, end_time_for_fit=50):
    """Plots the x and y positions of maximum topological charge over time, optionally including the linear fit."""
    plt.figure(figsize=(12, 6))
    plt.plot(times, x_positions, label='X Position', marker='o')
    
    if apply_linear_fit and best_fit_slope is not None and best_fit_intercept is not None:
        # Filter times to only include those within the specified fit range
        valid_times = [t for t in times if start_time_for_fit <= t <= end_time_for_fit]
        if valid_times:
            linear_fit = best_fit_slope * np.array(valid_times) + best_fit_intercept
            plt.plot(valid_times, linear_fit, label='Linear Fit', linestyle='--', color='red')
    
    plt.xlabel('Time [ns]', fontsize=22)
    plt.ylabel('Position [nm]', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(fontsize=20)
    plt.grid(True)
    plt.savefig(Path(output_directory) / f'positions_over_time_{file_suffix}.png', bbox_inches='tight')
    plt.close()


def plot_positions_with_two_fits(times, x_positions, y_positions, output_directory, file_suffix, apply_two_linear_fits=False, split_index=None, best_fit_slopes=None, best_fit_intercepts=None):
    """Plots the x and y positions of maximum topological charge over time, with optional two linear fits."""
    plt.figure(figsize=(12, 6))
    plt.plot(times, x_positions, label='X Position', marker='o')
    
    if apply_two_linear_fits and split_index is not None and best_fit_slopes is not None and best_fit_intercepts is not None:
        # Plot first linear fit (in red)
        linear_fit_1 = best_fit_slopes[0] * np.array(times[:split_index]) + best_fit_intercepts[0]
        plt.plot(times[:split_index], linear_fit_1, label='Linear Fit 1', linestyle='--', color='red')
        
        # Plot second linear fit (in orange)
        linear_fit_2 = best_fit_slopes[1] * np.array(times[split_index:]) + best_fit_intercepts[1]
        plt.plot(times[split_index:], linear_fit_2, label='Linear Fit 2', linestyle='--', color='orange')  # Use 'orange' for second fit
    
    # Set larger font sizes for labels, ticks, and title
    label_size = 20
    tick_size = 20 

    plt.xlabel('Time [ns]', fontsize=label_size)
    plt.ylabel('Position [nm]', fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.legend(fontsize=tick_size)
    plt.grid(True)
    plt.savefig(Path(output_directory) / f'positions_over_time_{file_suffix}.png', bbox_inches='tight')
    plt.close()

def plot_overall_topological_charge(times, overall_top_charges, output_directory, file_suffix):
    """Plots the overall topological charge over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(times, overall_top_charges, label='Overall Topological Charge', marker='o', color='blue')
    
    # Set larger font sizes for labels, ticks, and title
    label_size = 22 
    tick_size = 22 

    plt.xlabel('Time [ns]', fontsize=label_size)
    plt.ylabel('Topological Charge', fontsize=label_size)
    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)

    plt.grid(True)
    plt.savefig(Path(output_directory) / f'overall_topological_charge_{file_suffix}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    directory = '/cluster/home/mauricec/simulations/FeGe_5p5x5p5_10x4p5_res32_openBC_demag_uni010_run50ns.out'
    output_directory = '/cluster/home/mauricec/simulations'
    file_suffix = 'FeGe_5p5x5p5_10x4p5_res32_openBC_demag_uni010_run50ns'
    
    # Define the time interval you are interested in (e.g., 15ns to 25ns)
    time_start_ns = 0
    time_end_ns   = 50
    
    # Set the flags
    apply_linear_fit = True  # Flag for one linear fit
    apply_two_linear_fits = False  # Flag for two linear fits
    
    # Ensure only one of these flags is true
    if apply_linear_fit and apply_two_linear_fits:
        raise ValueError("Both 'apply_linear_fit' and 'apply_two_linear_fits' cannot be True simultaneously.")
    
    times, x_positions, y_positions, overall_top_charges = process_files(directory, time_start_ns=time_start_ns, time_end_ns=time_end_ns)
    
    if apply_two_linear_fits:
        split_index, best_fit_slopes, best_fit_intercepts, best_r_squared_total = find_two_linear_fits(times, x_positions)
        print(f"Split index = {split_index}")
        print(f"Best fit slopes (velocities): {best_fit_slopes[0]:.2f} nm/ns and {best_fit_slopes[1]:.2f} nm/ns with combined R^2 = {best_r_squared_total:.2f}")
        plot_positions_with_two_fits(times, x_positions, y_positions, output_directory, file_suffix, apply_two_linear_fits, split_index, best_fit_slopes, best_fit_intercepts)
    elif apply_linear_fit:
        best_fit_slope, best_fit_intercept, best_fit_r_squared = find_best_linear_fit(times, x_positions)
        print(f"Best fit slope (velocity) = {best_fit_slope:.2f} nm/ns with R^2 = {best_fit_r_squared:.2f}")
        plot_positions(times, x_positions, y_positions, output_directory, file_suffix, apply_linear_fit, best_fit_slope, best_fit_intercept)
    
    # Plot overall topological charge
    plot_overall_topological_charge(times, overall_top_charges, output_directory, file_suffix)
