import numpy as np
import matplotlib.pyplot as plt

# import colour
from uc3 import IMGS_REPORT, IMGS_STUDIO
from lib.histogram import compute_bins_count

imgs_studio = np.array(IMGS_STUDIO, dtype=object)
imgs_report = np.array(IMGS_REPORT, dtype=object)

# imgs_studio_hsv = np.empty(imgs_studio.shape, dtype=object)
# imgs_report_hsv = np.empty(imgs_report.shape, dtype=object)

# for i, img in enumerate(imgs_studio):
#     if len(img.shape) == 2:
#         # Components correction
#         img_rgb = np.empty((img.shape[0], img.shape[1], 3))
#         img_rgb[:, :, 0] = img
#         img_rgb[:, :, 1] = img
#         img_rgb[:, :, 2] = img
#         img = img_rgb
#     imgs_studio_hsv[i] = colour.RGB_to_HSV(img)
# for i, img in enumerate(imgs_report):
#     if len(img.shape) == 2:
#         # Components correction
#         img_rgb = np.empty((img.shape[0], img.shape[1], 3))
#         img_rgb[:, :, 0] = img
#         img_rgb[:, :, 1] = img
#         img_rgb[:, :, 2] = img
#         img = img_rgb
#     imgs_report_hsv[i] = colour.RGB_to_HSV(img)

# # Extract the RGB pixel values from each image and store them in a list
# pixel_values_studio = [img.reshape(-1, 3) for img in imgs_studio]
# pixel_values_studio = np.vstack(pixel_values_studio)

# pixel_values_report = [img.reshape(-1, 3) for img in imgs_report]
# pixel_values_report = np.vstack(pixel_values_report)

# # Extract the HSV pixel values from each image and store them in a list
# pixel_values_studio_hsv = [img.reshape(-1, 3) for img in imgs_studio_hsv]
# pixel_values_studio_hsv = np.vstack(pixel_values_studio_hsv)

# pixel_values_report_hsv = [img.reshape(-1, 3) for img in imgs_report_hsv]
# pixel_values_report_hsv = np.vstack(pixel_values_report_hsv)


def stack_images(imgs_origin):
    imgs = np.array(imgs_origin, dtype=object)

    for img in imgs:
        if len(img.shape) == 2:
            # Components correction
            img_rgb = np.empty((img.shape[0], img.shape[1], 3))
            img_rgb[:, :, 0] = img
            img_rgb[:, :, 1] = img
            img_rgb[:, :, 2] = img
            img = img_rgb

    # Extract the RGB pixel values from each image and store them in a list
    pixel_values = [img.reshape(-1, 3) for img in imgs]
    pixel_values = np.vstack(pixel_values)
    return pixel_values


def components_histograms(pixels, color_space: str, img_class: str):
    bins_count = compute_bins_count("sturges", pixels.shape[0])

    fig, axes = plt.subplots(1, 3)
    fig.suptitle(f"Pixel values histograms for {img_class} ({color_space})")
    fig.set_figheight(6)
    fig.set_figwidth(10)
    axes[0].set_title(f"{color_space[0]} component")
    axes[0].hist(pixels[:, 0], bins_count, density=True)
    axes[1].set_title(f"{color_space[1]} component")
    axes[1].hist(pixels[:, 1], bins_count, density=True)
    axes[2].set_title(f"{color_space[2]} component")
    axes[2].hist(pixels[:, 2], bins_count, density=True)
    plt.savefig(f"out/{img_class}_{color_space}_hist.png")
    plt.show()


imgs_studio_stacked = stack_images(IMGS_STUDIO)
imgs_report_stacked = stack_images(IMGS_REPORT)
components_histograms(imgs_studio_stacked, "RGB", "studio")
components_histograms(imgs_report_stacked, "RGB", "report")
