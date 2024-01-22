import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import imageio as iio
import scipy.ndimage
import seaborn as sea

def gaussian(sigma, x, mu=0):
    return (1 / sigma * np.sqrt(2 * np.pi)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )

def gaussian2D(sigma, x, y, mu=0):
    return (1 / sigma * np.sqrt(2 * np.pi)) * np.exp(
        -(((x - mu) ** 2) + ((y - mu) ** 2)) / (2 * sigma**2)
    )

def gaussian_prewitt(sigma, sz):
    h = (sz - 1) / 2
    x = np.linspace(-h, h, sz)
    y = x.T
    kx = np.zeros((sz, sz))
    ky = np.zeros((sz, sz))
    kx = kx + x
    ky = kx.T
    k = gaussian2D(sigma, kx, ky)
    k = k / np.sum(k)
    prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    gaussian_prewitt_x = scipy.ndimage.convolve(k, prewitt_x)
    gaussian_prewitt_y = scipy.ndimage.convolve(k, prewitt_y)
    return gaussian_prewitt_x, gaussian_prewitt_y

def apply_filter(img, filter):
    filter_size = len(filter)
    padding = filter_size // 2
    padded_image = np.pad(img, padding)
    filtered_image = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_image[i : i + filter_size, j : j + filter_size]
            filtered_image[i, j] = np.sum(region * filter)

    return filtered_image


img = rgb_to_gs(iio.imread("img/food_tea_03_tension.png").astype(np.uint8))

filter_x, filter_y = gaussian_prewitt(6, 9)

grad_x = apply_filter(img, filter_x)
grad_y = apply_filter(img, filter_y)

grad_complex = grad_x + 1j * grad_y

grad_norm = np.abs(grad_complex)
grad_ang = np.arctan(grad_x, grad_y)
grad_norm_flat = grad_norm.ravel()
grad_ang_flat = grad_ang.ravel()

print(np.min(grad_norm_flat), np.max(grad_norm_flat))
print(np.min(grad_ang_flat), np.max(grad_ang_flat))

bins_count = compute_bins_count("sturges", len(grad_norm_flat))
print("bins_count : ", bins_count)