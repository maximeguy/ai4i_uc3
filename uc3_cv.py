from uc3 import IMGS_REPORT, IMGS_STUDIO
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

imgs_studio = np.array(IMGS_STUDIO, dtype=object)
imgs_report = np.array(IMGS_REPORT, dtype=object)


def correct_components(img):
    # Image components correction
    img_corrected = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_corrected[:, :, 0] = img_corrected[:, :, 1] = img_corrected[:, :, 2] = img
    return img_corrected


def image_components_histograms(img, bins_count):
    """Compute histograms of an image components for feature extraction

    Args:
        img (_type_): _description_
        bins_count (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        list: Returns a list of the histograms of each component of an image
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input img should be a 3D RGB image (height, width, 3)")

    # Reshape image into a 1D pixel array
    img_reshaped = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
    # Compute marginal histograms
    hists = []
    for i in range(3):
        channel_hist = np.histogram(img_reshaped[:, i], bins=bins_count, density=True)[
            0
        ]
        hists.append(channel_hist)
    return hists


def extract_hist_features(hist):
    """Extract several features to qualify the distribution obtained from the histogram of an image component

    Args:
        hist (ndarray): The histogram on wich features are extracted

    Returns:
        ndarray: Array of features
    """

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


def classify(classifyer, classifyer_str):
    print(f"Classify with {classifyer_str}")
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


# Dataset containing all classes
imgs = np.concatenate([imgs_studio, imgs_report])

# First check images to determine a common bin count
ns = []
for img in imgs:
    ns.append(img.shape[0] * img.shape[1])
n = int(np.mean(ns))

# Extract RGB histograms of all images for feature extraction
imgs_features = []
for i in range(len(imgs)):
    img = imgs[i]
    if len(img.shape) != 3 or img.shape[2] != 3:
        img = correct_components(img)
    hists = image_components_histograms(img, n)

    components_features = []
    for hist in hists:
        component_feature = extract_hist_features(hist)
        for j in range(len(component_feature)):
            components_features.append(component_feature[j])
    imgs_features.append(components_features)
imgs_features = np.asarray(imgs_features)

labels = np.zeros(len(imgs_features))
labels[len(imgs_studio) :] = 1

X_train, X_test, y_train, y_test = train_test_split(
    imgs_features, labels, test_size=0.25, random_state=42
)

classify(KNeighborsClassifier(n_neighbors=9), "kNN (9)")
classify(GaussianNB(), "Naive Bayes")
