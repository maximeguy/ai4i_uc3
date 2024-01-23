import numpy as np
import matplotlib.pyplot as plt
from uc3 import IMGS_REPORT, IMGS_STUDIO
from lib.histogram import compute_bins_count, compute_bins_edges

imgs_studio = np.array(IMGS_STUDIO, dtype=object)
imgs_report = np.array(IMGS_REPORT, dtype=object)

# Extract the RGB pixel values from each image and store them in a list
pixel_values_studio = [img.reshape(-1, 3) for img in imgs_studio]
pixel_values_studio = np.vstack(pixel_values_studio)

pixel_values_report = [img.reshape(-1, 3) for img in imgs_report]
pixel_values_report = np.vstack(pixel_values_report)


def components_histograms(pixels, color_space: str, img_class: str):
    components_ranges = np.asarray(
        [
            [
                np.min(pixels[:, 0]),
                np.max(pixels[:, 0]),
            ],
            [
                np.min(pixels[:, 1]),
                np.max(pixels[:, 1]),
            ],
            [
                np.min(pixels[:, 2]),
                np.max(pixels[:, 2]),
            ],
        ],
        dtype=np.uint32,
    )
    bins_count = compute_bins_count("sturges", pixels.shape[0])
    bins_edges = compute_bins_edges(bins_count, components_ranges)

    fig, axes = plt.subplots(1, 3)
    fig.suptitle(f"Pixel values histograms for {img_class} ({color_space})")
    fig.set_figheight(6)
    fig.set_figwidth(10)
    axes[0].set_title(f"{color_space[0]} component")
    axes[0].hist(pixels[:, 0], bins_edges[0], density=True)
    axes[1].set_title(f"{color_space[1]} component")
    axes[1].hist(pixels[:, 1], bins_edges[1], density=True)
    axes[2].set_title(f"{color_space[2]} component")
    axes[2].hist(pixels[:, 2], bins_edges[2], density=True)
    plt.savefig(f"out/{img_class}_{color_space}_hist.png")
    plt.show()


components_histograms(pixel_values_studio, "RGB", "studio")
components_histograms(pixel_values_report, "RGB", "report")
