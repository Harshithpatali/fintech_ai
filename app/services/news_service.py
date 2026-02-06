import feedparser
import pandas as pd
from datetime import datetime
from app.config import RSS_SOURCES

class NewsService:

    @staticmethod
    def fetch_news():
        articles = []

        for url in RSS_SOURCES:
            feed = feedparser.parse(url)

            for entry in feed.entries:
                published = getattr(entry, "published", None)

                try:
                    date = datetime(*entry.published_parsed[:6]).date()
                except:
                    continue

                articles.append({
                    "date": date,
                    "title": entry.title,
                    "summary": entry.get("summary", "")
                })

        if not articles:
            return pd.DataFrame()

        df = pd.DataFrame(articles)
        return df
