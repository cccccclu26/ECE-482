# Stock Sentiment Analysis Project

## 项目结构

```
stock_sentiment/
├── .env                 # API密钥配置（不要上传到Git）
├── config.py            # 配置文件
├── news_fetcher.py      # Polygon新闻获取模块
├── sentiment_analyzer.py # LLM情绪分析模块
├── main.py              # 主程序入口
└── requirements.txt     # 依赖包列表
```

## 安装步骤

### 1. 创建虚拟环境
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置API密钥
将 `.env.example` 重命名为 `.env`，填入你的API密钥：
```
POLYGON_API_KEY=你的polygon密钥
OPENAI_API_KEY=你的openai密钥
```

### 4. 运行程序
```bash
python main.py
```

## 功能说明

1. **新闻获取**：从Polygon.io获取指定股票的近期新闻
2. **情绪分析**：使用GPT-4o-mini分析每条新闻的情绪
3. **评分聚合**：将多条新闻的情绪汇总为0-100的综合评分

## 注意事项

- 请勿将 `.env` 文件上传到公开仓库
- API调用会产生费用，请注意用量
