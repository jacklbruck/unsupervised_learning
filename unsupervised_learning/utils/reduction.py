from sklearn.random_projection import GaussianRandomProjection
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA, FastICA


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = None):
        self.clf = RandomForestClassifier()
        self.n_components = n_components

    def fit(self, *args, **kwargs):
        self.clf.fit(*args, **kwargs)

        return self

    def transform(self, X):
        mask = self.clf.feature_importances_.argsort()[::-1][: self.n_components]

        return X[:, mask]


class Identity(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = None):
        self.n_components = n_components

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X):
        return X


# Define component selections, manually set.
component_assignments = {
    "CC": {"ID": None, "ICA": 4, "PCA": 5, "RF": 5, "RP": 8},
    "PF": {"ID": None, "ICA": 7, "PCA": 7, "RF": 6, "RP": 9},
}

# Define base algorithms.
reductions = {
    "ID": Identity,
    "PCA": PCA,
    "ICA": FastICA,
    "RP": GaussianRandomProjection,
    "RF": FeatureSelector,
}
