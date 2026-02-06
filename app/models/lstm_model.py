import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Extensible LSTM for next-day return prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]

        output = self.fc(last_hidden)

        return output
