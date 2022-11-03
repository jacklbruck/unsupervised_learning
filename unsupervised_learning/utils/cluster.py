from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Define base algorithms.
algorithms = {
    "K-Means Clustering": KMeans,
    "Expectation Maximization": GaussianMixture,
}