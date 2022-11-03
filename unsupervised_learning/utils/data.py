import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from category_encoders import CatBoostEncoder
from openml.datasets import get_dataset

# Set dataset identifiers.
ids = {
    "customer-churn": 42178,
    "product-failures": 42890,
}

# Set target labels.
labels = {
    "customer-churn": "Churn",
    "product-failures": "Machine failure",
}


def load_dataset(k):
    X, y, _, _ = get_dataset(ids[k]).get_data(labels[k])

    if k == "customer-churn":
        X["TotalCharges"] = X["TotalCharges"].replace(" ", np.nan).astype("float")
        y = y.replace("Yes", 1).replace("No", 0)

        drop = []
        num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_features = list(set(X.columns) - set(num_features))

    if k == "product-failures":
        drop = ["UDI", "Product ID"]
        cat_features = ["Type", "TWF", "HDF", "PWF", "OSF", "RNF"]
        num_features = list(set(X.columns) - set(cat_features) - set(drop))

    # Encode categoricals.
    X = CatBoostEncoder(cols=cat_features).fit_transform(X, y)
    X[num_features] = (X[num_features] - X[num_features].mean()) / X[num_features].std()

    # Downsample.
    X, y = RandomUnderSampler(random_state=0).fit_resample(X, y)

    return X.drop(drop, axis=1), y


def load_datasets():
    return {k: load_dataset(k) for k in ids.keys()}
