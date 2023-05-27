import pandas as pd
from astropy.io import fits
import numpy as np

def prepare_data(path_images, csv_file):

    # Open the FITS file
    img = fits.open(path_images)  # 16
    img_data = img[0].data

    # NORMALIZE DATA
    img_data[img_data < -99] = 0
    img_data = img_data + 100
    img_data = np.log10(img_data)

    # ############# ROBUST ZSCORE

    # Calculate the mean and sd
    mean = 2.4232912
    sd = 0.25191346

    # Normalize the data using the median and IQR
    img_data = (img_data - mean) / sd

    # Read the source coordinates from the CSV file
    df = pd.read_csv(csv_file)
    source_coords = list(zip(df['x'].round().astype('int'), df['y'].round().astype('int')))

    # Create patches
    patches = []
    patch_size = 32
    for coord in source_coords:
        y, x = coord
        # Ensure the patch is fully within the image
        if x - patch_size // 2 >= 0 and y - patch_size // 2 >= 0 and x + patch_size // 2 < img_data.shape[0] and y + patch_size // 2 < img_data.shape[1]:
            patch = img_data[x - patch_size // 2 : x + patch_size // 2, y - patch_size // 2 : y + patch_size // 2]
            patches.append(patch)

    return patches