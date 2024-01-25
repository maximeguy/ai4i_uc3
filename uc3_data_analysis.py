import numpy as np
import matplotlib.pyplot as plt

# import colour
from uc3 import IMGS_REPORT, IMGS_STUDIO
from scipy.stats import skew, kurtosis
from lib.histogram import compute_bins_count
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

imgs_studio = np.array(IMGS_STUDIO, dtype=object)
imgs_report = np.array(IMGS_REPORT, dtype=object)


def correct_components(img):
    # Components correction
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_rgb[:, :, 0] = img
    img_rgb[:, :, 1] = img
    img_rgb[:, :, 2] = img
    return img_rgb


def stack_images(imgs_origin):
    imgs = np.array(imgs_origin, dtype=object)

    for img in imgs:
        if len(img.shape) == 2:
            img = correct_components(img)

    # Extract the RGB pixel values from each image and store them in a list
    pixel_values = [img.reshape(-1, 3) for img in imgs]
    pixel_values = np.vstack(pixel_values)
    return pixel_values


def components_histograms(pixels, color_space: str, img_class: str):
    bins_count = compute_bins_count("sturges", pixels.shape[0])

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.suptitle(f"Pixel values histograms for {img_class} ({color_space})")
    fig.set_figheight(6)
    fig.set_figwidth(12)
    axes[0].set_title(f"{color_space[0]} component")
    axes[0].hist(pixels[:, 0], bins_count, density=True, color="red")
    axes[0].set_xlabel("Pixel values")
    axes[0].set_ylabel("Occurrence probability density")
    axes[1].set_title(f"{color_space[1]} component")
    axes[1].hist(pixels[:, 1], bins_count, density=True, color="green")
    axes[2].set_title(f"{color_space[2]} component")
    axes[2].hist(pixels[:, 2], bins_count, density=True, color="blue")
    plt.savefig(f"out/{img_class}_{color_space}_hist.png")
    plt.show()


def histograms(image: np.array):
    h, w = image.shape[:2]
    bins_count = compute_bins_count("sturges", h * w)
    r_reshaped = image[:, :, 0].reshape((h * w))
    g_reshaped = image[:, :, 1].reshape((h * w))
    b_reshaped = image[:, :, 2].reshape((h * w))
    histo_red, bins = np.histogram(r_reshaped, bins=bins_count)
    histo_green, _ = np.histogram(g_reshaped, bins=bins_count)
    histo_blue, _ = np.histogram(b_reshaped, bins=bins_count)
    return histo_red, histo_green, histo_blue, bins


def image_components_histograms(img, bins_count):
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input img should be a 3D RGB image (height, width, 3)")

    img_reshaped = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))

    hists = []
    for i in range(3):
        channel_hist = plt.hist(img_reshaped[:, i], bins=bins_count, density=True)[0]
        hists.append(channel_hist)
    return hists


def extract_hist_features(hist):
    # Compute the mean
    mean = np.mean(hist)

    # Compute the median
    sorted_hist = np.sort(hist.flatten())
    median = np.median(sorted_hist)

    # Compute the standard deviation
    std_dev = np.std(hist)

    # Compute percentiles (e.g., 25th and 75th percentiles)
    percentile_25 = np.percentile(sorted_hist, 25)
    percentile_75 = np.percentile(sorted_hist, 75)

    mode = np.argmax(hist)

    entropy = -np.sum(
        hist * np.log2(hist + 1e-10)
    )  # Add a small constant to avoid log(0)

    variance = np.var(hist)

    features = [
        mean,
        median,
        std_dev,
        percentile_25,
        percentile_75,
        mode,
        skew(hist),
        kurtosis(hist),
        entropy,
        variance,
    ]
    return features


# Extract RGB histograms of all images for feature extraction
imgs = np.concatenate([imgs_studio, imgs_report])
imgs_features = []
for i in range(len(imgs)):
    img = imgs[i]
    # Correct images wich shapes differs from (w, h, 3)
    if len(img.shape) != 3 or img.shape[2] != 3:
        img = correct_components(img)
    hists = image_components_histograms(img, 18)

    components_features = []
    for hist in hists:
        component_feature = extract_hist_features(hist)
        for j in range(len(component_feature)):
            components_features.append(component_feature[j])
    imgs_features.append(components_features)
imgs_features = np.asarray(imgs_features)
print(imgs_features.shape)


def classify(classifyer, classifyer_str):
    clf = classifyer
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = clf.score(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot the confusion matrix
    # plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="viridis")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix [{classifyer_str}]")
    plt.savefig(f"out/cm_{classifyer_str}")
    # plt.show()


labels = np.zeros(len(imgs_studio) + len(imgs_report))
labels[len(imgs_studio) :] = 1

X_train, X_test, y_train, y_test = train_test_split(
    imgs_features, labels, test_size=0.3, random_state=42
)

classify(KNeighborsClassifier(n_neighbors=9), "kNN (9)")
classify(SVC(gamma="auto"), "SVM")
classify(GaussianNB(), "Naive Bayes")

# Generate components histograms of an array of all pixels of all the images of a class stacked together
# imgs_studio_stacked = stack_images(IMGS_STUDIO)
# imgs_report_stacked = stack_images(IMGS_REPORT)
# components_histograms(imgs_studio_stacked, "RGB", "studio")
# components_histograms(imgs_report_stacked, "RGB", "report")
