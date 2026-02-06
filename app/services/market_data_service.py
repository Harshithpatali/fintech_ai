import yfinance as yf
import pandas as pd
import time
from app.config import TICKER, START_DATE, END_DATE


class MarketDataService:

    MAX_RETRIES = 3

    @staticmethod
    def fetch_data():

        for attempt in range(MarketDataService.MAX_RETRIES):

            try:
                df = yf.download(
                    TICKER,
                    start=START_DATE,
                    end=END_DATE,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )

                if df is not None and not df.empty:

                    # Flatten MultiIndex if needed
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)

                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.index = pd.to_datetime(df.index)
                    df.sort_index(inplace=True)

                    return df

            except Exception as e:
                print(f"Market fetch attempt {attempt+1} failed:", e)

            time.sleep(2)

        raise ValueError("Market data fetch failed after retries.")
