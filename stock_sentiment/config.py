"""
配置文件 - 存放项目的各种配置参数
"""
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Polygon API 配置
POLYGON_BASE_URL = "https://api.polygon.io"

# 新闻获取配置
DEFAULT_NEWS_LIMIT = 10  # 每只股票默认获取的新闻数量
NEWS_LOOKBACK_DAYS = 3   # 获取过去几天的新闻

# LLM 配置
LLM_MODEL = "gpt-4o-mini"  # 使用的模型
LLM_TEMPERATURE = 0.3     # 较低的temperature使输出更稳定

# 目标股票列表（科技股）
TECH_STOCKS = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta
    "TSLA",   # Tesla
    "AMD",    # AMD
    "INTC",   # Intel
    "CRM",    # Salesforce
]

# 情绪分析 Prompt 模板
SENTIMENT_PROMPT = """你是一个专业的金融分析师。请分析以下关于{ticker}股票的新闻，并给出情绪评分。

新闻标题: {title}
新闻摘要: {description}
发布时间: {published_date}

请按以下格式返回JSON（不要包含其他内容）：
{{
    "sentiment": "bullish/neutral/bearish",
    "score": 0-100的整数（50为中性，100为极度看涨，0为极度看跌）,
    "confidence": 0-100的整数（表示你对这个判断的信心程度）,
    "reason": "简短的理由说明（不超过50字）"
}}

注意：
- 只返回JSON，不要有其他文字
- score必须是整数
- 分析时考虑新闻对股价的潜在影响
"""
