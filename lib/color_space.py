import numpy as np

def rgb_to_lab(img):
    """Convert RGB images to Lab and cast values as int

    Args:
        img (np.ndarray | str): Image can be passed as a path to fetch or an array

    Returns:
        np.ndarray: output
    """
    illuminant_RGB = np.array([0.31270, 0.32900])
    illuminant_XYZ = np.array([0.34570, 0.35850])
    matrix_RGB_to_XYZ = np.array(
        [
            [0.41240000, 0.35760000, 0.18050000],
            [0.21260000, 0.71520000, 0.07220000],
            [0.01930000, 0.11920000, 0.95050000],
        ]
    )

    RGB = np.asarray([])
    if img.__class__ == str:
        RGB = colour.read_image(img)
        print(RGB)
    else:
        RGB = img
    XYZ = colour.RGB_to_XYZ(
        RGB, illuminant_RGB, illuminant_XYZ, matrix_RGB_to_XYZ, "Bradford"
    )
    Lab = colour.XYZ_to_Lab(XYZ)
    return Lab


def rgb_to_gs(img):
    """Convert RGB image to grayscale

    Args:
        img (np.ndarray): Image input [[[R,G,B]]]

    Returns:
        np.ndarray: Image output [[GS]]
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
