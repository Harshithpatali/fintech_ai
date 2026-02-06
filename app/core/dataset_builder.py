import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from app.config import WINDOW_SIZE, TRAIN_SPLIT_RATIO
from app.config import WINDOW_SIZE


class TimeSeriesDatasetBuilder:
    """
    Builds sliding window dataset for LSTM.
    Handles:
    - Target shifting
    - Time-series safe split
    - Scaling (fit only on training data)
    - Tensor conversion
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.WINDOW_SIZE = WINDOW_SIZE
    def create_target(self, df):

        df = df.copy()
        df["target"] = (df["return"].shift(-1) > 0).astype(int)

        df.dropna(inplace=True)

        return df

    def create_sequences(self, data, target):

        X, y = [], []

        for i in range(len(data) - WINDOW_SIZE):
            X.append(data[i : i + WINDOW_SIZE])
            y.append(target[i + WINDOW_SIZE])

        return np.array(X), np.array(y)

    def train_test_split(self, X, y):

        split_index = int(len(X) * TRAIN_SPLIT_RATIO)

        X_train = X[:split_index]
        y_train = y[:split_index]

        X_test = X[split_index:]
        y_test = y[split_index:]

        return X_train, X_test, y_train, y_test

    def prepare(self, df):

        # 1️⃣ Create target
        df = self.create_target(df)

        feature_columns = [
            "return",
            "volatility",
            "hl_range",
            "log_volume",
            "sentiment",
        ]

        features = df[feature_columns].values
        target = df["target"].values

        # 2️⃣ Train/Test split BEFORE scaling (avoid leakage)
        split_index = int(len(features) * TRAIN_SPLIT_RATIO)

        train_features = features[:split_index]
        test_features = features[split_index:]

        # 3️⃣ Fit scaler ONLY on training
        self.scaler.fit(train_features)

        train_scaled = self.scaler.transform(train_features)
        test_scaled = self.scaler.transform(test_features)

        # 4️⃣ Combine back
        scaled_features = np.vstack([train_scaled, test_scaled])

        # 5️⃣ Create sliding windows
        X, y = self.create_sequences(scaled_features, target)

        # 6️⃣ Split sequences again (window shift reduces size)
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)

        # 7️⃣ Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        return X_train, X_test, y_train, y_test
