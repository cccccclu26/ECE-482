"""
主程序 - 股票情绪分析系统入口
"""
import argparse
from datetime import datetime
from typing import List, Dict
import pandas as pd

from news_fetcher import NewsFetcher
from sentiment_analyzer import SentimentAnalyzer
import config


class StockSentimentSystem:
    """股票情绪分析系统"""
    
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_stock(self, ticker: str, news_limit: int = 10) -> Dict:
        """
        分析单只股票的情绪
        
        Args:
            ticker: 股票代码
            news_limit: 获取的新闻数量
            
        Returns:
            完整的分析结果
        """
        print(f"\n{'='*60}")
        print(f"正在分析: {ticker}")
        print(f"{'='*60}")
        
        # 1. 获取新闻
        print(f"\n[1/2] 获取新闻...")
        news_list = self.news_fetcher.get_news(ticker, limit=news_limit)
        
        if not news_list:
            print(f"警告: 未找到 {ticker} 的相关新闻")
            return {
                "ticker": ticker,
                "analysis_time": datetime.now().isoformat(),
                "final_score": 50,
                "sentiment": "neutral",
                "news_count": 0,
                "message": "No news found"
            }
        
        print(f"  找到 {len(news_list)} 条新闻")
        
        # 2. 分析情绪
        print(f"\n[2/2] 分析情绪...")
        analysis_results = self.sentiment_analyzer.analyze_news_batch(news_list)
        
        # 3. 聚合结果
        aggregated = self.sentiment_analyzer.aggregate_sentiment(analysis_results)
        
        # 添加元数据
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
        分析多只股票
        
        Args:
            tickers: 股票列表，默认使用配置中的科技股
            news_limit: 每只股票获取的新闻数量
            
        Returns:
            包含所有股票分析结果的DataFrame
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
        """打印单只股票的分析结果"""
        print(f"\n{'='*60}")
        print(f"分析结果: {result['ticker']}")
        print(f"{'='*60}")
        print(f"最终评分: {result['final_score']:.1f} / 100")
        print(f"整体情绪: {result['sentiment'].upper()}")
        print(f"分析新闻数: {result['news_count']}")
        print(f"平均置信度: {result.get('avg_confidence', 0):.1f}%")
        print(f"看涨/中性/看跌: {result.get('bullish_count', 0)}/{result.get('neutral_count', 0)}/{result.get('bearish_count', 0)}")
        
        # 打印详细新闻分析
        if "details" in result and result["details"]:
            print(f"\n--- 新闻详情 ---")
            for i, detail in enumerate(result["details"], 1):
                sentiment_emoji = {
                    "bullish": "🟢",
                    "neutral": "🟡", 
                    "bearish": "🔴"
                }.get(detail.get("sentiment", "neutral"), "⚪")
                
                print(f"\n{i}. {sentiment_emoji} [{detail.get('score', 50)}分] {detail.get('title', 'N/A')[:50]}...")
                print(f"   理由: {detail.get('reason', 'N/A')}")
    
    def print_summary(self, df: pd.DataFrame):
        """打印多只股票的汇总表"""
        print(f"\n{'='*70}")
        print("股票情绪分析汇总")
        print(f"{'='*70}")
        print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"分析股票数: {len(df)}")
        print(f"\n{'='*70}")
        
        # 格式化输出表格
        print(f"{'排名':<4} {'股票':<8} {'评分':<8} {'情绪':<10} {'新闻数':<8} {'置信度':<8}")
        print("-" * 70)
        
        for i, row in df.iterrows():
            sentiment_display = {
                "bullish": "🟢 看涨",
                "neutral": "🟡 中性",
                "bearish": "🔴 看跌"
            }.get(row["sentiment"], "⚪ 未知")
            
            print(f"{i+1:<4} {row['ticker']:<8} {row['score']:<8.1f} {sentiment_display:<10} {row['news_count']:<8} {row['avg_confidence']:<8.1f}")
        
        print("-" * 70)
        
        # 统计信息
        bullish_stocks = df[df["sentiment"] == "bullish"]["ticker"].tolist()
        bearish_stocks = df[df["sentiment"] == "bearish"]["ticker"].tolist()
        
        if bullish_stocks:
            print(f"\n看涨股票: {', '.join(bullish_stocks)}")
        if bearish_stocks:
            print(f"看跌股票: {', '.join(bearish_stocks)}")
        
        print(f"\n平均评分: {df['score'].mean():.1f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="股票情绪分析系统")
    parser.add_argument(
        "-t", "--ticker",
        type=str,
        help="分析单只股票（如 AAPL）"
    )
    parser.add_argument(
        "-a", "--all",
        action="store_true",
        help="分析所有配置的科技股"
    )
    parser.add_argument(
        "-n", "--news-limit",
        type=int,
        default=5,
        help="每只股票获取的新闻数量（默认5）"
    )
    
    args = parser.parse_args()
    
    system = StockSentimentSystem()
    
    if args.ticker:
        # 分析单只股票
        result = system.analyze_stock(args.ticker.upper(), args.news_limit)
        system.print_result(result)
        
    elif args.all:
        # 分析所有股票
        df = system.analyze_multiple_stocks(news_limit=args.news_limit)
        system.print_summary(df)
        
    else:
        # 默认：演示模式，分析3只股票
        print("=" * 60)
        print("股票情绪分析系统 - Demo模式")
        print("=" * 60)
        print("\n使用方法:")
        print("  python main.py -t AAPL          # 分析单只股票")
        print("  python main.py -a               # 分析所有科技股")
        print("  python main.py -t NVDA -n 10    # 分析NVDA，获取10条新闻")
        print("\n现在运行Demo，分析 AAPL, NVDA, MSFT...")
        
        demo_stocks = ["AAPL", "NVDA", "MSFT"]
        df = system.analyze_multiple_stocks(tickers=demo_stocks, news_limit=3)
        system.print_summary(df)


if __name__ == "__main__":
    main()
