import cv2
import numpy as np
from pathlib import Path
import os
import subprocess


# Define the directories using pathlib for better path management
input_dir = Path("/cluster/home/mauricec/simulations/FeGe_7x7_20x5_res32_openBC_demag_init_track_pol_0p4_j010_80e10_30ns.out")
movies_dir = Path("/cluster/home/mauricec/simulations/movies")
movie_name = "FeGe_7x7_20x5_res32_openBC_demag_init_track_pol_0p4_j010_40e10_80ns.mp4"
framerate = 10

mumax3_convert_path = Path('/cluster/home/mauricec/mumax3.10_linux_cuda11.0')

def convert_ovf_to_png():
    """
    Converts all .ovf files in the input directory to .png format using mumax3-convert.
    Deletes any existing .png files to ensure all files are converted fresh.
    """
    os.environ['PATH'] += os.pathsep + str(mumax3_convert_path)
    for png_file in input_dir.glob('*.png'):
        png_file.unlink()
    
    for ovf_file in input_dir.glob('*.ovf'):
        print(f"Converting {ovf_file}...")
        subprocess.run(["mumax3-convert", "-png", str(ovf_file)], check=True)

def create_movie_from_png(framerate, movie_name):
    print("Creating movie from PNG files...")
    png_files = sorted(input_dir.glob('*.png'), key=lambda x: int(x.stem.split('m')[-1]))

    # Assuming all images are the same size, get dimensions of the first image
    first_image_path = str(png_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 output
    movie_path = movies_dir / movie_name
    video = cv2.VideoWriter(str(movie_path), fourcc, framerate, (width, height))

    for png_file in png_files:
        img = cv2.imread(str(png_file))
        video.write(img)  # Write the image to the video

    video.release()
    print(f"Movie created: {movie_path}")

def cleanup_png_files():
    """
    Deletes all .png files in the input directory.
    """
    for png_file in input_dir.glob('*.png'):
        png_file.unlink()
    print("All PNG files have been deleted.")

if __name__ == "__main__":
    convert_ovf_to_png()
    create_movie_from_png(framerate, movie_name)
    cleanup_png_files()
