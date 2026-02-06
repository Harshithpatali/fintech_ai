import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import numpy as np

from app.config import (
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    INPUT_SIZE,
)
from app.models.lstm_model import LSTMModel


class Trainer:

    def __init__(self, device="cpu"):
        self.device = device

        self.model = LSTMModel(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

    def train(self, X_train, y_train):

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False  # IMPORTANT: no shuffle in time-series
        )

        self.model.train()

        for epoch in range(EPOCHS):

            epoch_loss = 0

            for X_batch, y_batch in loader:

                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward pass
                predictions = self.model(X_batch)

                loss = self.criterion(predictions, y_batch)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

        return self.model

    def evaluate(self, X_test, y_test):

        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X_test.to(self.device)).cpu().numpy()

        y_true = y_test.cpu().numpy()

        rmse = np.sqrt(np.mean((predictions - y_true) ** 2))

        # Directional Accuracy
        direction_pred = np.sign(predictions)
        direction_true = np.sign(y_true)

        directional_accuracy = np.mean(direction_pred == direction_true)

        return predictions, rmse, directional_accuracy
