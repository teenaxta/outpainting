import os
import sys
import numpy as np

def add_white_borders(original_image, border_size_percentage=0.10):
    # Read the image

    # Get the dimensions of the original image
    height, width, _ = original_image.shape

    # Calculate the border size
    border_size = int(max(height, width) * border_size_percentage)

    # Create a new image with white borders
    new_height = height + 2 * border_size
    new_width = width + 2 * border_size
    new_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255  # 255 represents white color

    # Copy the original image to the center of the new image
    new_image[border_size:border_size + height, border_size:border_size + width] = original_image

    # Making mask
    mask = np.ones((new_height, new_width), dtype=np.uint8)*255 # 255 represents white color
    mask[border_size:-border_size, border_size:-border_size] = np.zeros_like(original_image[:,:,0])
    
    return new_image, mask, border_size

def get_image_path_from_cli():
    # Check if user provided an image path
    if len(sys.argv) != 2:
        print("Usage: python script.py <image_path>")
        sys.exit(1)

    # Get the image path from the command line argument
    image_path = sys.argv[1]

    # Check if the image path is valid
    if not os.path.exists(image_path):
        print("Error: Image path does not exist.")
        sys.exit(1)

    # Check if the provided path is a file
    if not os.path.isfile(image_path):
        print("Error: Path is not a file.")
        sys.exit(1)

    # Check if the file is an image (you can extend this check as needed)
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    if not image_path.lower().endswith(tuple(valid_image_extensions)):
        print("Error: File is not an image.")
        sys.exit(1)

    return image_path
