# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pyovf as OVF
from pathlib import Path
import skimage.filters
from scipy.signal import savgol_filter
import scipy.stats
import pandas as pd
from matplotlib.patches import Rectangle
from pprint import pprint

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

def determine_positions(file_paths, *, min_x_search, a, sigma=7):
    x_positions, y_positions = [], []

    min_x_index = round(min_x_search / a)        
    slice_x = slice(min_x_index, None)          

    for f in file_paths:
        data = read_ovf_data(f)                  
        field = skimage.filters.gaussian(data[0], sigma=sigma, channel_axis=2)

        tq = calculate_topological_charge_density(field)
        max_idx = np.unravel_index(np.argmax(tq[:, slice_x]), tq[:, slice_x].shape)

        y_positions.append(max_idx[0] * a)                       
        x_positions.append((max_idx[1] + min_x_index) * a)       

    return x_positions, y_positions

def evaluate_dislocation_position(x_pos, y_pos, d1, d3, d2=10 * 69.59e-9):
    wavelength = 69.59e-9
    y_center = d1/2
    y_min, y_max = y_center - wavelength / 4, y_center + wavelength / 4
    x_min, x_max = d1, d1 + d2 / 2

    in_y_init = y_min <= y_pos[0] <= y_max
    in_x_init = x_min <= x_pos[0] <= x_max
    in_y_final = y_min <= y_pos[1] <= y_max
    in_x_final = x_min <= x_pos[1] <= x_max

    if in_y_init and in_x_init:
        if in_y_final and in_x_final:
            category = "Centered, stable"
        else:
            category = "Centered, unstable"
    else:
        category = "Not Centered"

    return {
        "d1_lambda": d1 / wavelength,
        "d3_lambda": d3 / wavelength,
        "Category": category
    }

def analyze_geometry_set(base_dir: Path, csv_output: str):

    wavelength = 69.59e-9
    n_discre = 32
    step_size = n_discre // 2
    a = wavelength / n_discre

    index = 0
    results = []

    for i in range(1, 13):
        for j in range(0, i):
            nx1 = 3 * n_discre + i * step_size
            ny1 = 3 * n_discre + i * step_size
            nx2 = 10 * n_discre
            ny2 = 3 * n_discre + j * step_size

            d1 = nx1 * a
            d3 = ny2 * a

            f0 = base_dir / f"m{index:06}.ovf"
            f1 = base_dir / f"m{index+1:06}.ovf"
            index += 2

            if f0.exists() and f1.exists():
                x_pos, y_pos = determine_positions([f0, f1], min_x_search=d1, a=a)

                if len(x_pos) == 2:        
                    result = evaluate_dislocation_position(x_pos, y_pos, d1, d3)
                    results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(csv_output, index=False)
    print(f"Saved classification results to: {csv_output}")

def plot_geometry_classification(csv_path: str, output_path: str):

    df = pd.read_csv(csv_path)

    category_colors = {
        "Centered, stable": "#4CAF50",
        "Centered, unstable": "#FF9800",
        "Not Centered": "#F44336"
    }

    plt.figure(figsize=(10, 8))
    for category, color in category_colors.items():
        subset = df[df["Category"] == category]
        plt.scatter(subset["d1_lambda"], subset["d3_lambda"],
                    color=color, label=category, s=200, edgecolors='black')

    plt.xlabel(r"$d_1$ [$\lambda$]", fontsize=14)
    plt.ylabel(r"$d_3$ [$\lambda$]", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.title("Geometry Stability Classification", fontsize=16)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")



if __name__ == "__main__":

    directory = Path('/cluster/home/mauricec/simulations/geometry_analysis/geom_ana_lambda2_100ns_saveM_12.out')
    output_directory = Path('/cluster/home/mauricec/simulations/geometry_analysis/')
    file_suffix = 'geom_ana_lambda2_100ns_saveM_12'

    csv_output = output_directory / f"{file_suffix}_geometry_results.csv"
    plot_output = output_directory / f"{file_suffix}_geometry_plot.png"

    analyze_geometry_set(directory, csv_output)
    plot_geometry_classification(csv_output, plot_output)



