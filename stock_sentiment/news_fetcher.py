"""
News Fetcher Module - Retrieves stock news from Polygon.io
Supports both real-time and historical date range queries.
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import config


class NewsFetcher:
    """Fetch stock news from Polygon.io API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.POLYGON_API_KEY
        self.base_url = config.POLYGON_BASE_URL

        if not self.api_key:
            raise ValueError("Polygon API key is required. Please set POLYGON_API_KEY in .env file.")

    def get_news(
        self,
        ticker: str,
        limit: int = None,
        days_back: int = None
    ) -> List[Dict]:
        """
        Fetch recent news for a ticker (relative to today).

        Args:
            ticker: Stock ticker symbol
            limit: Max number of articles
            days_back: Lookback window in days

        Returns:
            List of processed news dicts
        """
        limit = limit or config.DEFAULT_NEWS_LIMIT
        days_back = days_back or config.NEWS_LOOKBACK_DAYS

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self.get_news_by_date(
            ticker,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            limit
        )

    def get_news_by_date(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        limit: int = None
    ) -> List[Dict]:
        """
        Fetch news for a ticker within a specific date range.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date string "YYYY-MM-DD"
            end_date: End date string "YYYY-MM-DD"
            limit: Max number of articles

        Returns:
            List of processed news dicts
        """
        limit = limit or config.DEFAULT_NEWS_LIMIT

        url = f"{self.base_url}/v2/reference/news"
        params = {
            "ticker": ticker,
            "published_utc.gte": start_date,
            "published_utc.lte": end_date,
            "limit": limit,
            "sort": "published_utc",
            "order": "desc",
            "apiKey": self.api_key
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "OK" or "results" in data:
                news_list = data.get("results", [])
                return self._process_news(news_list, ticker)
            else:
                print(f"  API error: {data}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"  News fetch failed: {e}")
            return []

    def _process_news(self, news_list: List[Dict], ticker: str) -> List[Dict]:
        """Clean and process raw news data"""
        processed = []

        for news in news_list:
            processed_news = {
                "ticker": ticker,
                "title": news.get("title", ""),
                "description": news.get("description", ""),
                "author": news.get("author", "Unknown"),
                "published_utc": news.get("published_utc", ""),
                "article_url": news.get("article_url", ""),
                "source": news.get("publisher", {}).get("name", "Unknown"),
            }

            if processed_news["title"] and processed_news["description"]:
                processed.append(processed_news)

        return processed


if __name__ == "__main__":
    fetcher = NewsFetcher()

    print("=" * 50)
    print("Test: Recent AAPL news")
    print("=" * 50)
    news = fetcher.get_news("AAPL", limit=3)
    for i, a in enumerate(news, 1):
        print(f"{i}. [{a['published_utc'][:10]}] {a['title'][:60]}")

    print("\n" + "=" * 50)
    print("Test: Historical AAPL news (Jan 2024)")
    print("=" * 50)
    news = fetcher.get_news_by_date("AAPL", "2024-01-01", "2024-01-07", limit=3)
    for i, a in enumerate(news, 1):
        print(f"{i}. [{a['published_utc'][:10]}] {a['title'][:60]}")
