import os
import imageio as iio
import numpy as np
import matplotlib.pyplot as plt

IMGS_STUDIO = []
IMGS_REPORT = []

def load_imgs(path, imgs):
    for filename in os.listdir(path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
            file_path = os.path.join(path, filename)
            img = iio.imread(file_path)
            imgs.append(img)

load_imgs("img/studio", IMGS_STUDIO)
load_imgs("img/report", IMGS_REPORT)
