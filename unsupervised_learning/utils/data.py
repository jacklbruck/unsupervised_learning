import numpy as np

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from category_encoders import CatBoostEncoder
from openml.datasets import get_dataset

# Set dataset identifiers.
ids = {
    "CC": 42178,
    "PF": 42890,
}

# Set target labels.
labels = {
    "CC": "Churn",
    "PF": "Machine failure",
}


def load_dataset(k):
    X, y, _, _ = get_dataset(ids[k]).get_data(labels[k])

    if k == "CC":
        X["TotalCharges"] = X["TotalCharges"].replace(" ", 0).astype("float")
        y = y.replace("Yes", 1).replace("No", 0)

        drop = []
        num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
        cat_features = list(set(X.columns) - set(num_features))

    if k == "PF":
        drop = ["UDI", "Product ID"]
        cat_features = ["Type", "TWF", "HDF", "PWF", "OSF", "RNF"]
        num_features = list(set(X.columns) - set(cat_features) - set(drop))

    # Drop features.
    X = X.drop(drop, axis=1).copy()

    # Downsample.
    X, y = RandomUnderSampler(random_state=0).fit_resample(X, y)

    # Split into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3)

    # Encode categoricals.
    enc = CatBoostEncoder(cols=cat_features).fit(X_train, y_train)

    # Transform features.
    X_train = enc.transform(X_train)
    X_test = enc.transform(X_test)

    X_test[num_features] = (X_test[num_features] - X_train[num_features].mean()) / X_train[num_features].std()
    X_train[num_features] = (X_train[num_features] - X_train[num_features].mean()) / X_train[num_features].std()

    return tuple(df.to_numpy() for df in (X_train, X_test, y_train, y_test))


def load_datasets():
    return {k: load_dataset(k) for k in ids.keys()}
