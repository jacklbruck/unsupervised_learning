import warnings
import logging

warnings.filterwarnings("ignore")

from .utils.reduction import reductions, component_assignments
from .utils.data import load_datasets
from .utils.cluster import clusters, cluster_assignments
from .utils import metrics
from pathlib import Path
from tqdm import tqdm
from json import dump


def run_clusters():
    data = load_datasets()

    for data_id in data.keys():
        logging.info(f"Running {data_id} dataset...")
        X_train, X_test, y_train, y_test = data[data_id]

        for reduction_id in reductions.keys():
            logging.info(f"\tRunning {reduction_id} reduction...")
            trf = reductions[reduction_id](n_components=component_assignments[data_id][reduction_id])

            if reduction_id == "RF":
                trf.fit(X_train, y_train)
            else:
                trf.fit(X_train)

            for cluster_id in clusters.keys():
                logging.info(f"\t\tRunning {cluster_id} cluster...")
                for n_clusters in tqdm(range(2, X_train.shape[1] + 1)):
                    # Set up container.
                    out = {}

                    # Initialize cluster.
                    clst = clusters[cluster_id](n_clusters=n_clusters).fit(trf.transform(X_train))

                    if cluster_id == "KM":
                        out.update({"SSE": clst.score(trf.transform(X_test))})

                    elif cluster_id == "EM":
                        out.update({"BIC": clst.bic(trf.transform(X_test))})

                    out.update(metrics.compute_cluster_statistics(trf.transform(X_test), y_test, clst))
                    out.update(
                        {
                            "Neural Network Testing Accuracy": metrics.compute_neural_network_test_accuracy(
                                X_train, X_test, y_train, y_test, trf, clst=clst
                            )
                        }
                    )
                    # Write output.
                    fd = (
                        Path(__file__).parents[1]
                        / "results"
                        / "cluster"
                        / data_id
                        / reduction_id
                        / cluster_id
                        / f"{n_clusters:02d}"
                    )
                    fd.mkdir(parents=True, exist_ok=True)

                    with open(fd / "out.json", "w+") as f:
                        dump(out, f)


def run_reductions():
    data = load_datasets()

    for data_id in data.keys():
        X_train, X_test, y_train, y_test = data[data_id]

        for reduction_id in reductions.keys():
            if reduction_id != "ICA":
                continue

            for n_components in range(1, X_train.shape[1] + 1):
                # Set up container.
                out = {}

                # Initialize reduction.
                trf = reductions[reduction_id](n_components=n_components)

                if reduction_id == "RF":
                    trf.fit(X_train, y_train)
                else:
                    trf.fit(X_train)

                if reduction_id == "ID":
                    out.update(
                        {
                            "Neural Network Testing Accuracy": metrics.compute_neural_network_test_accuracy(
                                X_train, X_test, y_train, y_test, trf
                            ),
                        }
                    )

                if reduction_id == "PCA":
                    out.update(
                        {
                            "Explained Variance Ratio": trf.explained_variance_ratio_.tolist(),
                            "Explained Variance": trf.explained_variance_ratio_.sum(),
                            "Neural Network Testing Accuracy": metrics.compute_neural_network_test_accuracy(
                                X_train, X_test, y_train, y_test, trf
                            ),
                        }
                    )

                elif reduction_id == "ICA":
                    out.update(
                        {
                            "Kurtosis": metrics.compute_kurtosis(X_test, trf),
                            "Neural Network Testing Accuracy": metrics.compute_neural_network_test_accuracy(
                                X_train, X_test, y_train, y_test, trf
                            ),
                        }
                    )

                elif reduction_id == "RP":
                    out.update(
                        {
                            "Pairwise Distance Correlation": metrics.compute_pairwise_distance_correlation(X_test, trf),
                            "Reconstruction Error": metrics.compute_reconstruction_error(X_test, trf),
                            "Neural Network Testing Accuracy": metrics.compute_neural_network_test_accuracy(
                                X_train, X_test, y_train, y_test, trf
                            ),
                        }
                    )

                elif reduction_id == "RF":
                    mask = trf.clf.feature_importances_.argsort()[-n_components:]

                    out.update(
                        {
                            "Feature Importance Ratio": trf.clf.feature_importances_[mask].tolist(),
                            "Feature Importance": trf.clf.feature_importances_[mask].sum(),
                            "Neural Network Testing Accuracy": metrics.compute_neural_network_test_accuracy(
                                X_train, X_test, y_train, y_test, trf
                            ),
                        }
                    )

                # Write output.
                fd = Path(__file__).parents[1] / "results" / "reduction" / data_id / reduction_id / f"{n_components:02d}"
                fd.mkdir(parents=True, exist_ok=True)

                with open(fd / "out.json", "w+") as f:
                    dump(out, f)


def main():
    run_reductions()
    # run_clusters()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
