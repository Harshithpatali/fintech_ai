import pandas as pd
import numpy as np

from app.services.market_data_service import MarketDataService
from app.services.news_service import NewsService
from app.services.sentiment_service import SentimentService
from app.config import VOL_WINDOW
from app.core.wavelet_transformer import WaveletTransformer


class DataPipeline:
    """
    Orchestrates:
    - Market data ingestion
    - Feature engineering
    - News ingestion
    - Sentiment aggregation
    - Data cleaning
    - Wavelet denoising
    """

    def run(self) -> pd.DataFrame:

        # =========================
        # 1️⃣ Fetch Market Data
        # =========================
        market_df = MarketDataService.fetch_data()

        # =========================
        # 2️⃣ Derived Financial Features
        # =========================
        market_df["return"] = market_df["Close"].pct_change()
        market_df["volatility"] = market_df["return"].rolling(VOL_WINDOW).std()
        market_df["hl_range"] = (
            (market_df["High"] - market_df["Low"]) / market_df["Close"]
        )
        market_df["log_volume"] = np.log1p(market_df["Volume"])

        market_df.dropna(inplace=True)

        # =========================
        # 3️⃣ News Ingestion
        # =========================
        news_df = NewsService.fetch_news()

        sentiment_service = SentimentService()
        sentiment_df = sentiment_service.score_dataframe(news_df)

        # =========================
        # 4️⃣ Merge Sentiment
        # =========================
        if not sentiment_df.empty:
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
            sentiment_df.set_index("date", inplace=True)

            market_df = market_df.merge(
                sentiment_df,
                left_index=True,
                right_index=True,
                how="left"
            )

        # Fill missing sentiment with neutral
        market_df["sentiment"].fillna(0, inplace=True)

        # =========================
        # 5️⃣ Remove Incomplete Sessions
        # =========================
        market_df = market_df[market_df["Volume"] > 0]

        # =========================
        # 6️⃣ Haar Wavelet Denoising
        # =========================
        wavelet = WaveletTransformer(level=1)

        columns_to_denoise = ["return", "Close"]

        market_df = wavelet.transform(
            df=market_df,
            columns=columns_to_denoise
        )

        # =========================
        # 7️⃣ Final Cleanup
        # =========================
        market_df.dropna(inplace=True)
        market_df.sort_index(inplace=True)

        return market_df
