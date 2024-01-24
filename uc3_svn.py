import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from uc3 import IMGS_REPORT, IMGS_STUDIO
from lib.histogram import compute_bins_count
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_dataset(imgs_origin):
    imgs = np.array(imgs_origin, dtype=object)
    for i, img in enumerate(imgs):
        if len(img.shape) == 2:
            # Components correction
            img_rgb = np.empty((img.shape[0], img.shape[1], 3))
            img_rgb[:, :, 0] = img
            img_rgb[:, :, 1] = img
            img_rgb[:, :, 2] = img
            img = img_rgb
        imgs[i] = img
    return imgs


def images_histograms(imgs, bins_count):
    hists = np.zeros((len(imgs), bins_count**3))
    for img in imgs:
        plt.imshow(img)
        plt.show()
        img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        hist = np.histogramdd(img, bins_count, density=True, range=[(0,255),(0,255),(0,255)])[0]
        np.append(hists, hist)
    return hists


imgs_studio = prepare_dataset(IMGS_STUDIO)
imgs_report = prepare_dataset(IMGS_REPORT)[: len(imgs_studio)]

# ns = np.zeros(len(imgs_studio)+ len(imgs_report))
# for i, img in enumerate(np.concatenate(imgs_studio, imgs_report)):
#         ns[i] = img.shape[0] * img.shape[1]
# bins_count = compute_bins_count("sturges", np.median(ns))
hists_studio = images_histograms(imgs_studio, 18)
hists_report = images_histograms(imgs_report, 18)
labels = np.zeros(len(imgs_studio) * 2)
labels[len(imgs_studio) :] = 1

hists = np.concatenate([hists_studio, hists_report])
print(hists)
features = np.zeros((len(hists), 5))
for i, hist in enumerate(hists):
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

    features[i] = [mean, median, std_dev, percentile_25, percentile_75]
print(features)
# imgs = np.concatenate([imgs_studio, imgs_report])
# for i,img in enumerate(imgs):
#     print(img.shape, labels[i])
print(hists.shape, labels.shape)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
clf = SVC(gamma="auto")
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot the confusion matrix using Seaborn for better visualization
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="g")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
