import sys
import os

# Ensure project root is in path (Windows-safe)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import mlflow

from app.pipelines.data_pipeline import DataPipeline
from app.core.walk_forward import WalkForwardEngine
from app.core.backtester import Backtester


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # =====================================
    # 1️⃣ Data Ingestion
    # =====================================
    print("Running data pipeline...")
    df = DataPipeline().run()

    # =====================================
    # 2️⃣ Walk-Forward Training
    # =====================================
    print("Starting walk-forward training...")

    engine = WalkForwardEngine(
        train_years=5,
        test_years=1,
        device=device
    )

    wf_df = engine.run(df)

    # =====================================
    # 3️⃣ Backtesting
    # =====================================
    print("Running backtest...")

    backtester = Backtester()

    backtest_df = backtester.run(
        wf_df,
        wf_df["prediction"].values
    )

    strategy_sharpe = backtester.sharpe_ratio(
        backtest_df["strategy_return"]
    )

    market_sharpe = backtester.sharpe_ratio(
        backtest_df["return"]
    )

    # =====================================
    # 4️⃣ MLflow Logging
    # =====================================
    mlflow.set_experiment("FinSight_NIFTY50_WalkForward")

    with mlflow.start_run():

        mlflow.log_param("train_years", 5)
        mlflow.log_param("test_years", 1)

        mlflow.log_metric("Strategy_Sharpe", strategy_sharpe)
        mlflow.log_metric("Market_Sharpe", market_sharpe)

    # =====================================
    # 5️⃣ Final Output
    # =====================================
    print("\n==============================")
    print("Walk-Forward Results")
    print("==============================")
    print("Strategy Sharpe:", round(strategy_sharpe, 4))
    print("Market Sharpe:  ", round(market_sharpe, 4))
    print("==============================\n")


if __name__ == "__main__":
    main()
