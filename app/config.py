from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Market
TICKER = "^NSEI"
START_DATE = "2010-01-01"
END_DATE = None  # fetch till today

# Volatility
VOL_WINDOW = 20

# News
NEWS_KEYWORDS = "NIFTY OR NSE OR Indian stock market"
RSS_SOURCES = [
    "https://news.google.com/rss/search?q=NIFTY+50",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://www.reuters.com/markets/asia/rss"
]

# Sentiment
FINBERT_MODEL = "ProsusAI/finbert"

# Sliding Window
WINDOW_SIZE = 30

# Train/Test
TRAIN_SPLIT_RATIO = 0.8
# Model Hyperparameters
INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
EPOCHS = 30
BATCH_SIZE = 32
