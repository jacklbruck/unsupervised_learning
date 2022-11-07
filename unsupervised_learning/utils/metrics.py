import numpy as np

from sklearn.metrics import homogeneity_score, completeness_score, adjusted_mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import kurtosis


def compute_neural_network_test_accuracy(X_train, X_test, y_train, y_test, trf, clst=None):
    # Initialize model.
    clf = MLPClassifier(hidden_layer_sizes=(8, 8), solver="lbfgs")

    # Train model.
    if clst is not None:
        # Set up cluster features model.
        C_train = clst.predict(trf.transform(X_train)).reshape(-1)
        C_test = clst.predict(trf.transform(X_test)).reshape(-1)

        # Extract number of clusters.
        n_clusters = max([C_train.max(), C_test.max()]) + 1

        # Train model.
        clf.fit(np.concatenate([trf.transform(X_train), np.eye(n_clusters)[C_train]], axis=1), y_train)

        # Apply model.
        y_pred = clf.predict(np.concatenate([trf.transform(X_test), np.eye(n_clusters)[C_test]], axis=1))
    else:
        # Train model.
        clf.fit(trf.transform(X_train), y_train)

        # Apply model.
        y_pred = clf.predict(trf.transform(X_test))

    return accuracy_score(y_test, y_pred)


def compute_kurtosis(X_test, trf):
    return np.median(np.abs(kurtosis(trf.transform(X_test), axis=0)))


def compute_pairwise_distance_correlation(X_test, trf):
    return np.corrcoef(pairwise_distances(X_test).flatten(), pairwise_distances(trf.transform(X_test)).flatten())[0, 1]


def compute_reconstruction_error(X_test, trf):
    return np.nanmean(np.power(X_test - ((np.linalg.pinv(trf.components_) @ trf.components_) @ (X_test.T)).T, 2))


def compute_cluster_statistics(X_test, y_test, clst):
    return {
        "Homogeneity Score": homogeneity_score(y_test, clst.predict(X_test)),
        "Completeness Score": completeness_score(y_test, clst.predict(X_test)),
        "Adjusted Mutual Information Score": adjusted_mutual_info_score(y_test, clst.predict(X_test)),
    }
