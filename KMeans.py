import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as euclidean
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

penguins = pd.read_csv(
    "https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")

penguins = penguins.dropna()
# counts = penguins.species_short.value_counts()
Adelies = penguins.loc[penguins["species_short"] == "Adelie"]
Gentoos = penguins.loc[penguins["species_short"] == "Gentoo"]
Chinstraps = penguins.loc[penguins["species_short"] == "Chinstrap"]

X_train_dataframe = pd.concat([Adelies.iloc[:100, :], Gentoos.iloc[:80, :], Chinstraps.iloc[:50, :]])
X_test_dataframe = pd.concat([Adelies.iloc[100:, :], Gentoos.iloc[80:, :], Chinstraps.iloc[50:, :]])


def init_centroid(X_data, k):
    idx = np.random.choice(len(X_data), k, replace=False)
    centroids = X_data[idx, :]
    return centroids


def assign_samples(X_data, centroids):
    distances = euclidean(X_data, centroids)
    points = np.array([np.argmin(i) for i in distances])
    return points


def centroid_calc(X_data, k, threshold, iterations):
    n = len(X_data)
    initial_centroids = init_centroid(X_data, k)
    points = assign_samples(X_data, initial_centroids)
    for _ in range(iterations):
        centroids = []
        for idx in range(k):
            temp_cent = X_data[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Updated Centroids

        distances = euclidean(X_data, centroids)
        points = np.array([np.argmin(i) for i in distances])

        E = (1 / n) * np.sum(distances)
        if E < threshold:
            break
    return points, E


def k_means(X_data, k, threshold, iterations):
    labels, E = centroid_calc(X_data, k, threshold, iterations)
    return np.array(labels), E


X_train = X_train_dataframe[
    [
        "culmen_length_mm",
        "flipper_length_mm",
    ]
].values

y_train_labels_vector = X_train_dataframe.species_short.map({"Adelie": 0, "Gentoo": 1, "Chinstrap": 2})
y_train = y_train_labels_vector.to_numpy()

labels, E = k_means(X_train, 3, 0.03, 10000)
u_labels = np.unique(labels)

for i in u_labels:
    plt.scatter(X_train[labels == i, 0], X_train[labels == i, 1], label=i)
plt.legend(["Adelie", "Gentoo", "Chinstrap"])
plt.show()

X_train = X_train_dataframe[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values

labels, E = k_means(X_train, 3, 0.003, 10000)
u_labels = np.unique(labels)

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
sns.pairplot(X_train_dataframe, hue='species_short')
plt.show()
