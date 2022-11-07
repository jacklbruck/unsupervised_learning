from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Define cluster selections, manually set.
cluster_assignments = {
    "PF": {
        "KM": {"ID": 8, "ICA": 6, "PCA": 8, "RF": 7, "RP": 7},
        "EM": {"ID": 9, "ICA": 7, "PCA": 9, "RF": 9, "RP": 8},
    },
    "CC": {
        "KM": {"ID": 8, "ICA": 7, "PCA": 8, "RF": 10, "RP": 9},
        "EM": {"ID": 10, "ICA": 8, "PCA": 10, "RF": 12, "RP": 10},
    },
}


# Define base algorithms.
clusters = {
    "KM": KMeans,
    "EM": lambda n_clusters: GaussianMixture(n_components=n_clusters),
}
