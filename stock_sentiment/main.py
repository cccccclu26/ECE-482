"""
Main entry point - Stock Sentiment Analysis System
"""
import argparse
from datetime import datetime
from typing import List, Dict
import pandas as pd

from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
import config


class StockSentimentSystem:
    """Stock Sentiment Analysis System"""

    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()

    def analyze_stock(self, ticker: str, news_limit: int = 10) -> Dict:
        """
        Analyze sentiment for a single stock.

        Args:
            ticker: Stock ticker symbol
            news_limit: Number of news articles to fetch

        Returns:
            Complete analysis result
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {ticker}")
        print(f"{'='*60}")

        # 1. Fetch news
        print(f"\n[1/2] Fetching news...")
        news_list = self.news_fetcher.get_news(ticker, limit=news_limit)

        if not news_list:
            print(f"Warning: No news found for {ticker}")
            return {
                "ticker": ticker,
                "analysis_time": datetime.now().isoformat(),
                "final_score": 50,
                "sentiment": "neutral",
                "news_count": 0,
                "message": "No news found"
            }

        print(f"  Found {len(news_list)} articles")

        # 2. Analyze sentiment
        print(f"\n[2/2] Analyzing sentiment...")
        analysis_results = self.sentiment_analyzer.analyze_news_batch(news_list)

        # 3. Aggregate results
        aggregated = self.sentiment_analyzer.aggregate_sentiment(analysis_results)

        result = {
            "ticker": ticker,
            "analysis_time": datetime.now().isoformat(),
            **aggregated
        }

        return result

    def analyze_multiple_stocks(
        self,
        tickers: List[str] = None,
        news_limit: int = 5
    ) -> pd.DataFrame:
        """
        Analyze multiple stocks.

        Args:
            tickers: List of tickers, defaults to config.TECH_STOCKS
            news_limit: Number of news articles per stock

        Returns:
            DataFrame with all analysis results
        """
        tickers = tickers or config.TECH_STOCKS
        results = []

        for ticker in tickers:
            result = self.analyze_stock(ticker, news_limit)
            results.append({
                "ticker": result["ticker"],
                "score": result["final_score"],
                "sentiment": result["sentiment"],
                "news_count": result["news_count"],
                "avg_confidence": result.get("avg_confidence", 0),
                "bullish": result.get("bullish_count", 0),
                "bearish": result.get("bearish_count", 0),
                "neutral": result.get("neutral_count", 0),
            })

        df = pd.DataFrame(results)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        return df

    def print_result(self, result: Dict):
        """Print analysis result for a single stock"""
        print(f"\n{'='*60}")
        print(f"Result: {result['ticker']}")
        print(f"{'='*60}")
        print(f"Final Score:    {result['final_score']:.1f} / 100")
        print(f"Sentiment:      {result['sentiment'].upper()}")
        print(f"Articles:       {result['news_count']}")
        print(f"Avg Confidence: {result.get('avg_confidence', 0):.1f}%")
        print(f"Bullish/Neutral/Bearish: {result.get('bullish_count', 0)}/{result.get('neutral_count', 0)}/{result.get('bearish_count', 0)}")

        if "details" in result and result["details"]:
            print(f"\n--- Article Details ---")
            for i, detail in enumerate(result["details"], 1):
                sentiment_tag = {
                    "bullish": "[+]",
                    "neutral": "[=]",
                    "bearish": "[-]"
                }.get(detail.get("sentiment", "neutral"), "[?]")

                title = detail.get('title', 'N/A')[:60]
                print(f"\n{i}. {sentiment_tag} [Score:{detail.get('score', 50)}] {title}")
                print(f"   Reason: {detail.get('reason', 'N/A')}")

    def print_summary(self, df: pd.DataFrame):
        """Print summary table for multiple stocks"""
        print(f"\n{'='*70}")
        print("Stock Sentiment Analysis Summary")
        print(f"{'='*70}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stocks Analyzed: {len(df)}")
        print(f"\n{'='*70}")

        print(f"{'Rank':<6} {'Ticker':<8} {'Score':<8} {'Sentiment':<12} {'Articles':<10} {'Confidence':<10}")
        print("-" * 70)

        for i, row in df.iterrows():
            sentiment_display = {
                "bullish": "BULLISH",
                "neutral": "NEUTRAL",
                "bearish": "BEARISH"
            }.get(row["sentiment"], "UNKNOWN")

            print(f"{i+1:<6} {row['ticker']:<8} {row['score']:<8.1f} {sentiment_display:<12} {row['news_count']:<10} {row['avg_confidence']:<10.1f}")

        print("-" * 70)

        bullish_stocks = df[df["sentiment"] == "bullish"]["ticker"].tolist()
        bearish_stocks = df[df["sentiment"] == "bearish"]["ticker"].tolist()

        if bullish_stocks:
            print(f"\nBullish: {', '.join(bullish_stocks)}")
        if bearish_stocks:
            print(f"Bearish: {', '.join(bearish_stocks)}")

        print(f"\nAverage Score: {df['score'].mean():.1f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stock Sentiment Analysis System")
    parser.add_argument(
        "-t", "--ticker",
        type=str,
        help="Analyze a single stock (e.g., AAPL)"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="Analyze all configured tech stocks"
    )
    parser.add_argument(
        "-n", "--news-limit",
        type=int,
        default=5,
        help="Number of news articles per stock (default: 5)"
    )

    args = parser.parse_args()

    system = StockSentimentSystem()

    if args.ticker:
        result = system.analyze_stock(args.ticker.upper(), args.news_limit)
        system.print_result(result)

    elif args.all:
        df = system.analyze_multiple_stocks(news_limit=args.news_limit)
        system.print_summary(df)

    else:
        print("=" * 60)
        print("Stock Sentiment Analysis System")
        print("=" * 60)
        print("\nUsage:")
        print("  python main.py -t AAPL          # Analyze single stock")
        print("  python main.py -a               # Analyze all tech stocks")
        print("  python main.py -t NVDA -n 10    # Analyze NVDA with 10 articles")
        print("\nRunning demo with AAPL, NVDA, MSFT...")

        demo_stocks = ["AAPL", "NVDA", "MSFT"]
        df = system.analyze_multiple_stocks(tickers=demo_stocks, news_limit=3)
        system.print_summary(df)


if __name__ == "__main__":
    main()
