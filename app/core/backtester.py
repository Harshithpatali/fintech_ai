import numpy as np
import pandas as pd


class Backtester:
    """
    Professional backtesting engine with:
    - Volatility targeting
    - Clipped exposure
    - Equity curve generation
    - Sharpe ratio calculation
    """

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    def run(self, df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """
        Runs volatility-targeted strategy.
        """

        df = df.copy()

        # Align predictions with last N rows
        df = df.iloc[-len(predictions):].copy()

        df["prediction"] = predictions.flatten()

        # =========================
        # Volatility Targeting
        # =========================

        # Avoid division by zero
        df["volatility"] = df["volatility"].replace(0, 1e-8)

        df["signal"] = df["prediction"] / df["volatility"]

        # Cap leverage
        df["signal"] = np.clip(df["signal"], -1, 1)

        # =========================
        # Strategy Returns
        # =========================

        df["strategy_return"] = df["signal"] * df["return"]

        # =========================
        # Equity Curves
        # =========================

        df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
        df["market_equity"] = (1 + df["return"]).cumprod()

        return df

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Computes annualized Sharpe ratio.
        """

        excess_returns = returns - (self.risk_free_rate / 252)

        if excess_returns.std() == 0:
            return 0.0

        sharpe = (
            np.mean(excess_returns)
            / np.std(excess_returns)
        ) * np.sqrt(252)

        return sharpe
