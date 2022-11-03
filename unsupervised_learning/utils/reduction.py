from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA, KernelPCA, FastICA

# Define base algorithms.
algorithms = {
    "Principal Component Analysis": PCA,
    "Independent Component Analysis": FastICA,
    "Random Projection": GaussianRandomProjection,
    "Kernel Principal Component Analysis": KernelPCA,
}