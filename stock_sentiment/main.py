"""
Main entry point - Stock Sentiment & Technical Analysis System

Final Score (0-100) = Sentiment (60%) + Technical (40%)
"""
import argparse
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd

from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
from technical_analysis import TechnicalAnalyzer
from combined_scorer import combine_scores
import config

# ML scorer (optional - used with --ml flag)
try:
    from ml_scorer import MLScorer
    ML_AVAILABLE = True
except (ImportError, FileNotFoundError):
    ML_AVAILABLE = False

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ensure_results_dir():
    """Create results directory if it doesn't exist"""
    os.makedirs(RESULTS_DIR, exist_ok=True)


class StockAnalysisSystem:
    """Stock Analysis System - Combines Sentiment + Technical Analysis"""

    def __init__(self, use_ml: bool = False):
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.use_ml = use_ml
        self.ml_scorer = None

        if use_ml:
            if not ML_AVAILABLE:
                print("WARNING: ML scorer not available. Run 'python ml_scorer.py --train' first.")
                print("Falling back to rule-based scoring.")
                self.use_ml = False
            else:
                try:
                    self.ml_scorer = MLScorer()
                    print("[ML Mode] Using trained Logistic Regression model for scoring")
                except FileNotFoundError:
                    print("WARNING: No trained model found. Run 'python ml_scorer.py --train' first.")
                    print("Falling back to rule-based scoring.")
                    self.use_ml = False

    def analyze_stock(self, ticker: str, news_limit: int = 20) -> Dict:
        """
        Full analysis for a single stock: sentiment + technical -> final score.

        Args:
            ticker: Stock ticker symbol
            news_limit: Number of news articles to fetch

        Returns:
            Complete analysis result
        """
        print(f"\n{'='*60}")
        print(f"Analyzing: {ticker}")
        print(f"{'='*60}")

        # ---- Sentiment Analysis (60%) ----
        print(f"\n[1/3] Sentiment Analysis (weight: 60%)")
        print(f"  Fetching news (last {config.NEWS_LOOKBACK_DAYS} days, limit {news_limit})...")
        news_list = self.news_fetcher.get_news(ticker, limit=news_limit)

        if news_list:
            print(f"  Found {len(news_list)} articles")
            analysis_results = self.sentiment_analyzer.analyze_news_batch(news_list)
            sentiment_agg = self.sentiment_analyzer.aggregate_sentiment(analysis_results)
            sentiment_score = sentiment_agg["final_score"]
        else:
            print(f"  No news found, using neutral (50)")
            sentiment_agg = {
                "final_score": 50, "sentiment": "neutral", "news_count": 0,
                "avg_confidence": 0, "bullish_count": 0, "bearish_count": 0,
                "neutral_count": 0, "details": []
            }
            sentiment_score = 50

        # ---- Technical Analysis (40%) ----
        print(f"\n[2/3] Technical Analysis (weight: 40%)")
        try:
            tech_result = self.technical_analyzer.analyze(ticker)
            technical_score = tech_result["technical_score"]
        except Exception as e:
            print(f"  Technical analysis failed: {e}, using neutral (50)")
            tech_result = {
                "technical_score": 50, "signal": "neutral",
                "ema": {}, "rsi": {}, "price": {}
            }
            technical_score = 50

        # ---- Combined Score ----
        print(f"\n[3/3] Calculating combined score...")

        if self.use_ml and self.ml_scorer:
            ml_result = self.ml_scorer.predict(
                sentiment_score=sentiment_score,
                sentiment_confidence=sentiment_agg.get("avg_confidence"),
                news_count=sentiment_agg.get("news_count", 0),
                technical_data=tech_result,
            )
            final_score = ml_result["ml_score"]
            grade = ml_result["grade"]
            breakdown = {
                "sentiment_score": round(sentiment_score, 2),
                "technical_score": round(technical_score, 2),
                "scoring_method": "ml_logistic_regression",
                "probability_up": ml_result["probability_up"],
                "probability_down": ml_result["probability_down"],
            }
        else:
            combined = combine_scores(sentiment_score, technical_score)
            final_score = combined["final_score"]
            grade = combined["grade"]
            breakdown = combined["breakdown"]

        result = {
            "ticker": ticker,
            "analysis_time": datetime.now().isoformat(),
            "lookback_days": config.NEWS_LOOKBACK_DAYS,
            "final_score": final_score,
            "grade": grade,
            "breakdown": breakdown,
            "sentiment": sentiment_agg,
            "technical": tech_result,
        }

        return result

    def analyze_multiple_stocks(
        self,
        tickers: List[str] = None,
        news_limit: int = 20
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
        all_results = []
        rows = []

        for ticker in tickers:
            result = self.analyze_stock(ticker, news_limit)
            all_results.append(result)
            rows.append({
                "ticker": result["ticker"],
                "final_score": result["final_score"],
                "grade": result["grade"],
                "sentiment_score": result["breakdown"]["sentiment_score"],
                "technical_score": result["breakdown"]["technical_score"],
                "news_count": result["sentiment"]["news_count"],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)

        # Save results
        self.save_results(all_results, df)

        return df

    def save_results(self, full_results: List[Dict], summary_df: pd.DataFrame):
        """Save analysis results to files"""
        ensure_results_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(RESULTS_DIR, f"analysis_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False, default=str)

        csv_path = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
        summary_df.to_csv(csv_path, index=False)

        print(f"\n[Saved] Detailed -> {json_path}")
        print(f"[Saved] Summary  -> {csv_path}")

    def save_single_result(self, result: Dict):
        """Save a single stock analysis result"""
        ensure_results_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = result.get("ticker", "UNKNOWN")

        json_path = os.path.join(RESULTS_DIR, f"{ticker}_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n[Saved] {ticker} -> {json_path}")

    def print_result(self, result: Dict):
        """Print analysis result for a single stock"""
        bd = result["breakdown"]
        sent = result["sentiment"]
        tech = result.get("technical", {})

        print(f"\n{'='*60}")
        print(f"  {result['ticker']}  |  FINAL SCORE: {result['final_score']} / 100  |  {result['grade']}")
        print(f"{'='*60}")

        if bd.get("scoring_method") == "ml_logistic_regression":
            print(f"\n  Scoring Method: ML Logistic Regression")
            print(f"  Sentiment Input:  {bd['sentiment_score']:.1f}")
            print(f"  Technical Input:  {bd['technical_score']:.1f}")
            print(f"  P(Up): {bd['probability_up']:.2%}  P(Down): {bd['probability_down']:.2%}")
        else:
            print(f"\n  Sentiment:  {bd['sentiment_score']:.1f} x {bd['sentiment_weight']:.0%} = {bd['sentiment_contribution']:.1f} pts")
            print(f"  Technical:  {bd['technical_score']:.1f} x {bd['technical_weight']:.0%} = {bd['technical_contribution']:.1f} pts")

        # Sentiment details
        print(f"\n  --- Sentiment ({sent.get('news_count', 0)} articles) ---")
        print(f"  Bullish: {sent.get('bullish_count', 0)}  Neutral: {sent.get('neutral_count', 0)}  Bearish: {sent.get('bearish_count', 0)}")
        print(f"  Avg Confidence: {sent.get('avg_confidence', 0):.1f}%")

        # Technical details
        ema = tech.get("ema", {})
        rsi = tech.get("rsi", {})
        price = tech.get("price", {})

        if ema:
            print(f"\n  --- Technical (EMA + RSI) ---")
            print(f"  EMA Score: {ema.get('score', 'N/A')}  (raw: {ema.get('raw_points', 'N/A')}/4)")
            print(f"  EMA100 Uptrend: {ema.get('ema100_uptrend', 'N/A')}")
            print(f"  EMA50 > EMA100: {ema.get('ema50_above_ema100', 'N/A')}")
            print(f"  EMA25 > EMA100: {ema.get('ema25_above_ema100', 'N/A')}")

        if rsi:
            print(f"  RSI: {rsi.get('value', 'N/A')}  ({rsi.get('signal', 'N/A')})  Score: {rsi.get('score', 'N/A')}")

        if price:
            print(f"\n  --- Price ---")
            print(f"  Current: ${price.get('current', 'N/A')}  EMA25: ${price.get('ema25', 'N/A')}  EMA50: ${price.get('ema50', 'N/A')}  EMA100: ${price.get('ema100', 'N/A')}")

        # Top news
        details = sent.get("details", [])
        if details:
            print(f"\n  --- Top News ---")
            for i, d in enumerate(details[:5], 1):
                tag = {"bullish": "[+]", "neutral": "[=]", "bearish": "[-]"}.get(d.get("sentiment", ""), "[?]")
                print(f"  {i}. {tag} [{d.get('score', 50)}] {d.get('title', 'N/A')[:55]}")

    def print_summary(self, df: pd.DataFrame):
        """Print summary table for multiple stocks"""
        print(f"\n{'='*75}")
        print("Stock Analysis Summary  (Final = Sentiment x 60% + Technical x 40%)")
        print(f"{'='*75}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Lookback: {config.NEWS_LOOKBACK_DAYS} days  |  Stocks: {len(df)}")
        print(f"\n{'Rank':<5} {'Ticker':<8} {'Final':<8} {'Grade':<18} {'Sent.':<8} {'Tech.':<8} {'News':<6}")
        print("-" * 75)

        for i, row in df.iterrows():
            print(f"{i+1:<5} {row['ticker']:<8} {row['final_score']:<8.1f} {row['grade']:<18} {row['sentiment_score']:<8.1f} {row['technical_score']:<8.1f} {row['news_count']:<6}")

        print("-" * 75)
        print(f"\nAvg Final Score: {df['final_score'].mean():.1f}")

        buys = df[df["final_score"] >= 60]["ticker"].tolist()
        sells = df[df["final_score"] <= 40]["ticker"].tolist()
        if buys:
            print(f"Bullish: {', '.join(buys)}")
        if sells:
            print(f"Bearish: {', '.join(sells)}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Stock Sentiment & Technical Analysis System")
    parser.add_argument("-t", "--ticker", type=str, help="Analyze a single stock (e.g., AAPL)")
    parser.add_argument("-a", "--all", action="store_true", help="Analyze all configured tech stocks")
    parser.add_argument("-n", "--news-limit", type=int, default=20, help="News articles per stock (default: 20)")
    parser.add_argument("--ml", action="store_true", help="Use ML (Logistic Regression) scoring instead of rule-based")

    args = parser.parse_args()
    system = StockAnalysisSystem(use_ml=args.ml)

    if args.ticker:
        result = system.analyze_stock(args.ticker.upper(), args.news_limit)
        system.print_result(result)
        system.save_single_result(result)

    elif args.all:
        df = system.analyze_multiple_stocks(news_limit=args.news_limit)
        system.print_summary(df)

    else:
        print("=" * 60)
        print("Stock Analysis System (Sentiment 60% + Technical 40%)")
        print("=" * 60)
        print("\nUsage:")
        print("  python main.py -t AAPL          # Analyze single stock")
        print("  python main.py -a               # Analyze all tech stocks")
        print("  python main.py -t NVDA -n 10    # Custom article count")
        print("\nRunning demo with AAPL, NVDA, MSFT...")

        demo_stocks = ["AAPL", "NVDA", "MSFT"]
        df = system.analyze_multiple_stocks(tickers=demo_stocks, news_limit=20)
        system.print_summary(df)


if __name__ == "__main__":
    main()
