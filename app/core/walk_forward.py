import numpy as np
import pandas as pd
import torch

from app.core.dataset_builder import TimeSeriesDatasetBuilder
from app.core.trainer import Trainer


class WalkForwardEngine:

    def __init__(self, train_years=5, test_years=1, device="cpu"):
        self.train_years = train_years
        self.test_years = test_years
        self.device = device

    def run(self, df):

        df = df.copy()
        df["year"] = df.index.year

        unique_years = sorted(df["year"].unique())

        all_predictions = []
        all_dates = []

        for i in range(len(unique_years)):

            train_start = unique_years[i]
            train_end = train_start + self.train_years - 1
            test_end = train_end + self.test_years

            if test_end > unique_years[-1]:
                break

            train_df = df[
                (df["year"] >= train_start) &
                (df["year"] <= train_end)
            ].copy()

            test_df = df[
                (df["year"] > train_end) &
                (df["year"] <= test_end)
            ].copy()

            if len(train_df) < 200 or len(test_df) < 50:
                continue

            print(f"Training {train_start}-{train_end} | Testing {train_end+1}-{test_end}")

            # =========================
            # TRAIN DATA PREPARATION
            # =========================

            builder = TimeSeriesDatasetBuilder()

            train_df = builder.create_target(train_df)

            feature_cols = [
                "return",
                "volatility",
                "hl_range",
                "log_volume",
                "sentiment",
            ]

            X_train_np = train_df[feature_cols].values
            y_train_np = train_df["target"].values

            builder.scaler.fit(X_train_np)
            X_train_scaled = builder.scaler.transform(X_train_np)

            X_train, y_train = builder.create_sequences(
                X_train_scaled,
                y_train_np
            )

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

            # =========================
            # TRAIN MODEL
            # =========================

            trainer = Trainer(device=self.device)
            trainer.train(X_train, y_train)

            # =========================
            # TEST DATA PREPARATION
            # =========================

            test_df = builder.create_target(test_df)

            X_test_np = test_df[feature_cols].values
            y_test_np = test_df["target"].values

            X_test_scaled = builder.scaler.transform(X_test_np)

            X_test, _ = builder.create_sequences(
                X_test_scaled,
                y_test_np
            )

            if len(X_test) == 0:
                continue

            X_test = torch.tensor(X_test, dtype=torch.float32)

            # =========================
            # PREDICT
            # =========================

            trainer.model.eval()
            with torch.no_grad():
                preds = trainer.model(X_test).cpu().numpy().flatten()

            # Align predictions with correct dates
            prediction_dates = test_df.index[builder.WINDOW_SIZE:]

            all_predictions.extend(preds)
            all_dates.extend(prediction_dates)

        result_df = df.loc[all_dates].copy()
        result_df["prediction"] = all_predictions

        return result_df
