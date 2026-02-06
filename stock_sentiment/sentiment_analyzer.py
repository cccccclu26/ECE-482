"""
情绪分析模块 - 使用OpenAI GPT分析新闻情绪
"""
import json
from typing import List, Dict, Optional
from openai import OpenAI
import config


class SentimentAnalyzer:
    """使用LLM分析新闻情绪"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
    
    def analyze_single_news(self, news: Dict) -> Dict:
        """
        分析单条新闻的情绪
        
        Args:
            news: 新闻字典，包含ticker, title, description, published_utc
            
        Returns:
            情绪分析结果，包含sentiment, score, confidence, reason
        """
        # 构建prompt
        prompt = config.SENTIMENT_PROMPT.format(
            ticker=news.get("ticker", "Unknown"),
            title=news.get("title", ""),
            description=news.get("description", ""),
            published_date=news.get("published_utc", "")
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的金融分析师，擅长分析新闻对股票的影响。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            # 解析返回的JSON
            result_text = response.choices[0].message.content.strip()
            
            # 尝试提取JSON（处理可能的markdown代码块）
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            # 添加原始新闻信息
            result["title"] = news.get("title", "")
            result["source"] = news.get("source", "")
            result["published_utc"] = news.get("published_utc", "")
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始返回: {result_text}")
            return self._default_result(news)
            
        except Exception as e:
            print(f"分析失败: {e}")
            return self._default_result(news)
    
    def _default_result(self, news: Dict) -> Dict:
        """返回默认的中性结果"""
        return {
            "sentiment": "neutral",
            "score": 50,
            "confidence": 0,
            "reason": "分析失败，使用默认值",
            "title": news.get("title", ""),
            "source": news.get("source", ""),
            "published_utc": news.get("published_utc", "")
        }
    
    def analyze_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        """
        批量分析多条新闻
        
        Args:
            news_list: 新闻列表
            
        Returns:
            分析结果列表
        """
        results = []
        
        for i, news in enumerate(news_list, 1):
            print(f"  分析第 {i}/{len(news_list)} 条新闻...")
            result = self.analyze_single_news(news)
            results.append(result)
        
        return results
    
    def aggregate_sentiment(self, results: List[Dict]) -> Dict:
        """
        聚合多条新闻的情绪评分
        
        使用置信度加权平均计算最终得分
        
        Args:
            results: 情绪分析结果列表
            
        Returns:
            聚合后的情绪评分，包含final_score, sentiment, news_count等
        """
        if not results:
            return {
                "final_score": 50,
                "sentiment": "neutral",
                "news_count": 0,
                "avg_confidence": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0
            }
        
        # 统计各类情绪数量
        bullish_count = sum(1 for r in results if r.get("sentiment") == "bullish")
        bearish_count = sum(1 for r in results if r.get("sentiment") == "bearish")
        neutral_count = sum(1 for r in results if r.get("sentiment") == "neutral")
        
        # 置信度加权平均
        total_weight = 0
        weighted_score = 0
        
        for r in results:
            confidence = r.get("confidence", 50)
            score = r.get("score", 50)
            
            # 使用置信度作为权重
            weight = confidence / 100.0 if confidence > 0 else 0.5
            weighted_score += score * weight
            total_weight += weight
        
        # 计算最终得分
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 50
        
        # 根据得分确定整体情绪
        if final_score >= 60:
            overall_sentiment = "bullish"
        elif final_score <= 40:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"
        
        # 平均置信度
        avg_confidence = sum(r.get("confidence", 50) for r in results) / len(results)
        
        return {
            "final_score": round(final_score, 2),
            "sentiment": overall_sentiment,
            "news_count": len(results),
            "avg_confidence": round(avg_confidence, 2),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "details": results  # 保留详细结果
        }


# 测试代码
if __name__ == "__main__":
    # 模拟新闻数据进行测试
    test_news = [
        {
            "ticker": "AAPL",
            "title": "Apple Reports Record iPhone Sales in Q4",
            "description": "Apple Inc. announced record-breaking iPhone sales for the fourth quarter, exceeding analyst expectations by 15%. The company's services revenue also grew significantly.",
            "published_utc": "2024-01-15T10:00:00Z",
            "source": "Reuters"
        },
        {
            "ticker": "AAPL",
            "title": "Apple Faces Supply Chain Challenges in China",
            "description": "Apple is experiencing production delays at its major manufacturing facilities in China due to ongoing supply chain disruptions.",
            "published_utc": "2024-01-14T08:00:00Z",
            "source": "Bloomberg"
        }
    ]
    
    analyzer = SentimentAnalyzer()
    
    print("=" * 50)
    print("测试情绪分析")
    print("=" * 50)
    
    results = analyzer.analyze_news_batch(test_news)
    
    for r in results:
        print(f"\n标题: {r['title']}")
        print(f"情绪: {r['sentiment']} | 评分: {r['score']} | 置信度: {r['confidence']}")
        print(f"理由: {r['reason']}")
    
    # 测试聚合
    print("\n" + "=" * 50)
    print("聚合结果")
    print("=" * 50)
    
    aggregated = analyzer.aggregate_sentiment(results)
    print(f"最终评分: {aggregated['final_score']}")
    print(f"整体情绪: {aggregated['sentiment']}")
    print(f"分析新闻数: {aggregated['news_count']}")
