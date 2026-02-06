import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import FINBERT_MODEL

class SentimentService:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        self.model.eval()

    def score_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment_score = probs[0][2] - probs[0][0]  # positive - negative

        return sentiment_score.item()

    def score_dataframe(self, news_df):

        if news_df.empty:
            return pd.DataFrame()

        news_df["sentiment"] = news_df.apply(
            lambda row: self.score_text(row["title"] + " " + row["summary"]),
            axis=1
        )

        daily_sentiment = news_df.groupby("date")["sentiment"].mean().reset_index()

        return daily_sentiment
