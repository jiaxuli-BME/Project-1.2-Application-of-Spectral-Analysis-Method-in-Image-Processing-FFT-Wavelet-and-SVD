import numpy as np
from matplotlib.image import imread

def load_image(image_path):
    image = imread(image_path)
    gray_image = np.mean(image, -1)  # Convert to grayscale
    return gray_image
