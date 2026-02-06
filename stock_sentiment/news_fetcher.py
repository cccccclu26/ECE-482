"""
新闻获取模块 - 从Polygon.io获取股票相关新闻
"""
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import config


class NewsFetcher:
    """从Polygon.io获取股票新闻"""
    
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
        获取指定股票的新闻
        
        Args:
            ticker: 股票代码，如 "AAPL"
            limit: 返回新闻数量上限
            days_back: 获取过去几天的新闻
            
        Returns:
            新闻列表，每条新闻包含title, description, published_utc等字段
        """
        limit = limit or config.DEFAULT_NEWS_LIMIT
        days_back = days_back or config.NEWS_LOOKBACK_DAYS
        
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # 构建API请求
        url = f"{self.base_url}/v2/reference/news"
        params = {
            "ticker": ticker,
            "published_utc.gte": start_date.strftime("%Y-%m-%d"),
            "published_utc.lte": end_date.strftime("%Y-%m-%d"),
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
                print(f"API返回异常: {data}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"获取新闻失败: {e}")
            return []
    
    def _process_news(self, news_list: List[Dict], ticker: str) -> List[Dict]:
        """处理和清洗新闻数据"""
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
            
            # 跳过没有标题或描述的新闻
            if processed_news["title"] and processed_news["description"]:
                processed.append(processed_news)
        
        return processed
    
    def get_multiple_stocks_news(
        self, 
        tickers: List[str] = None,
        limit_per_stock: int = None
    ) -> Dict[str, List[Dict]]:
        """
        获取多只股票的新闻
        
        Args:
            tickers: 股票代码列表
            limit_per_stock: 每只股票获取的新闻数量
            
        Returns:
            字典，key为股票代码，value为新闻列表
        """
        tickers = tickers or config.TECH_STOCKS
        all_news = {}
        
        for ticker in tickers:
            print(f"正在获取 {ticker} 的新闻...")
            news = self.get_news(ticker, limit=limit_per_stock)
            all_news[ticker] = news
            print(f"  获取到 {len(news)} 条新闻")
        
        return all_news


# 测试代码
if __name__ == "__main__":
    fetcher = NewsFetcher()
    
    # 测试获取单只股票新闻
    print("=" * 50)
    print("测试获取 AAPL 新闻")
    print("=" * 50)
    
    news = fetcher.get_news("AAPL", limit=5)
    
    for i, article in enumerate(news, 1):
        print(f"\n--- 新闻 {i} ---")
        print(f"标题: {article['title']}")
        print(f"来源: {article['source']}")
        print(f"时间: {article['published_utc']}")
        print(f"摘要: {article['description'][:100]}...")
