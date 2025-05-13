import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle


viridis = plt.get_cmap('viridis')
seismic = plt.get_cmap('seismic')
coolwarm = plt.get_cmap('coolwarm')

def rgba_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(
        int(color[0]*255), int(color[1]*255), int(color[2]*255)
    )

def plot_geometry_classification(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    df = df[df["d1_lambda"] <= 8]

    category_colors = {
        "Centered, stable":     rgba_to_hex(viridis(0.9)),
        "Centered, unstable":   rgba_to_hex(viridis(0.7)),
        "Off-centered, stable": rgba_to_hex(viridis(0.4)),
        "Off-centered, unstable": rgba_to_hex(viridis(0.1))
    }

    plt.figure(figsize=(10, 8))
    gap = 0.02  # small spacing between rectangles

    for category, color in category_colors.items():
        subset = df[df["Category"] == category]
        for _, row in subset.iterrows():
            x = row["d1_lambda"]
            y = row["d3_lambda"]
            grid_spacing = 0.5
            rect = Rectangle((x - grid_spacing/2 + gap/2, y - grid_spacing/2 + gap/2),
                            grid_spacing - gap, grid_spacing - gap,
                            facecolor=color, edgecolor='none')
            plt.gca().add_patch(rect)


    # Create ticks based on unique sorted values
    xticks = sorted(df["d1_lambda"].unique())
    yticks = sorted(df["d3_lambda"].unique())

    plt.xticks(xticks, fontsize=14)
    plt.yticks(yticks, fontsize=14)

    plt.xlim(min(xticks) - 0.3, max(xticks) + 0.3)
    plt.ylim(min(yticks) - 0.3, max(yticks) + 0.3)

    plt.xlabel(r"$d_1$ [$\lambda$]", fontsize=14)
    plt.ylabel(r"$d_3$ [$\lambda$]", fontsize=14)

    legend_elements = [
        Patch(facecolor=color, label=label)
        for label, color in category_colors.items()
    ]

    plt.legend(handles=legend_elements, loc='upper left', fontsize=14, frameon=True)


    plt.gca().xaxis.set_major_locator(MultipleLocator(0.5))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
    plt.grid(True)
    
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    input_csv = Path("csv_data") / "geom_ana_lambda2_100ns_saveM_12_results.csv"
    plot_output = Path("plots") / "geometry_plot.png"
    plot_geometry_classification(input_csv, plot_output)
