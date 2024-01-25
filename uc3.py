import os
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt

IMGS_STUDIO = []
IMGS_REPORT = []

def correct_components(img):
    # Components correction
    img_corrected = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_corrected[:, :, 0] = img_corrected[:, :, 1] = img_corrected[:, :, 2] = img
    return img_corrected

def load_imgs(path, imgs):
    for filename in os.listdir(path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
            file_path = os.path.join(path, filename)
            img = iio.imread(file_path).astype(np.uint8)
            imgs.append(img)
    for img in imgs:
        if len(img.shape) != 3 or img.shape[2] != 3:
            img = correct_components(img)

    return imgs


load_imgs("img/studio", IMGS_STUDIO)
load_imgs("img/report", IMGS_REPORT)
