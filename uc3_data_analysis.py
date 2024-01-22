import numpy as np
import imageio as iio
import os

folder_path = "/img"  # Replace with the path to your folder containing images
imgs_studio = []
imgs_report = []

# Loop through all files in the folder
for filename in os.listdir(folder_path + "studio"):
    if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
        file_path = os.path.join(folder_path, filename)
        img = iio.imread(file_path)
        imgs_studio.append(img)

for filename in os.listdir(folder_path + "report"):
    if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
        file_path = os.path.join(folder_path, filename)
        img = iio.imread(file_path)
        imgs_report.append(img)
