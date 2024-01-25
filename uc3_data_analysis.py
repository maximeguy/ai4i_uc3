import numpy as np
import matplotlib.pyplot as plt
from uc3 import IMGS_REPORT, IMGS_STUDIO

def compute_bins_count(method: str, n: int):
    """Compute the number of bins from the selected method for n elements

    Args:
        method (str): _description_
        n (int): _description_

    Returns:
        _type_: _description_
    """
    if method == "sturges":
        bins_count = 1 + np.log2(n)
    elif method == "rice":
        bins_count = 2 * np.cbrt(n)
    elif method == "sqrt":
        bins_count = np.sqrt(n)
    return int(bins_count)

def stack_images(imgs_origin):
    """Stack images of a dataset into a 1D pixel array

    Args:
        imgs_origin (ndarray): Original images array

    Returns:
        ndarray: The concatenated pixel array
    """
    imgs = np.array(imgs_origin, dtype=object)
    # Extract the pixels values from each image and store them in a list
    pixel_values = [img.reshape(-1, 3) for img in imgs]
    return np.vstack(pixel_values)


def components_histograms(pixels, color_space: str, img_class: str):
    """Compute histograms of RGB components of a pixel array for vizualization

    Args:
        pixels (ndarray): pixel array
        color_space (str): _description_
        img_class (str): _description_
    """
    bins_count = compute_bins_count("sturges", pixels.shape[0])

    fig, axes = plt.subplots(1, pixels.shape[1], sharex=True, sharey=True)
    fig.suptitle(f"Pixel values histograms for {img_class} ({color_space})")
    fig.set_figheight(6)
    fig.set_figwidth(12)

    axes[0].set_xlabel("Pixel values")
    axes[0].set_ylabel("Occurrence probability density")

    for i in range(3):
        axes[i].set_title(f"{color_space[i]} component")
        axes[i].hist(pixels[:, i], bins_count, density=True, color="red")

    plt.savefig(f"out/{img_class}_{color_space}_hist.png")

# Generate components histograms of an array of all pixels of all the images of a class stacked together
imgs_studio_stacked = stack_images(IMGS_STUDIO)
imgs_report_stacked = stack_images(IMGS_REPORT)
components_histograms(imgs_studio_stacked, "RGB", "studio")
components_histograms(imgs_report_stacked, "RGB", "report")
