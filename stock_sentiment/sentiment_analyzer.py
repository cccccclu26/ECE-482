"""
Sentiment Analysis Module - Dual-model ensemble using WaveSpeed AI API.
Runs both Claude 3.7 Sonnet and GPT-5 on each article, averages their scores.
Supports concurrent LLM calls for faster batch analysis.
"""
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import config


class SentimentAnalyzer:
    """Analyze news sentiment using dual-LLM ensemble via WaveSpeed AI API"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.WAVESPEED_API_KEY

        if not self.api_key:
            raise ValueError("WaveSpeed API key is required. Please set WAVESPEED_API_KEY in .env file.")

        self.api_url = config.WAVESPEED_API_URL
        self.models = config.LLM_MODELS
        self.max_workers = config.MAX_CONCURRENT_LLM_CALLS

    def _call_llm(self, prompt: str, model: str) -> str:
        """
        Call WaveSpeed AI API with a specific model.

        Args:
            prompt: The prompt to send
            model: Model identifier (e.g., "anthropic/claude-3.7-sonnet")

        Returns:
            Response text from the LLM
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "enable_sync_mode": True,
            "model": model,
            "priority": "latency",
            "prompt": prompt,
            "reasoning": False
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        data = response.json()

        if data.get("code") == 200 and data.get("data", {}).get("outputs"):
            return data["data"]["outputs"][0]

        raise RuntimeError(f"LLM API error ({model}): {data.get('message', 'Unknown error')}")

    def _parse_llm_response(self, result_text: str) -> Dict:
        """Parse LLM response text into a dict, handling markdown fences."""
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        return json.loads(result_text)

    def _analyze_with_single_model(self, prompt: str, model: str) -> Dict:
        """Run analysis with one model, return parsed result or None on failure."""
        try:
            raw = self._call_llm(prompt, model)
            result = self._parse_llm_response(raw)
            model_short = model.split("/")[-1]
            result["model"] = model_short
            return result
        except Exception as e:
            model_short = model.split("/")[-1]
            print(f"    {model_short} failed: {e}")
            return None

    def _average_model_results(self, results: List[Dict]) -> Dict:
        """Average scores from multiple model results."""
        valid = [r for r in results if r is not None]
        if not valid:
            return None

        avg_score = round(sum(r.get("score", 50) for r in valid) / len(valid))
        avg_confidence = round(sum(r.get("confidence", 50) for r in valid) / len(valid))

        # Determine sentiment from averaged score
        if avg_score >= 60:
            sentiment = "bullish"
        elif avg_score <= 40:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # Combine reasons
        reasons = [f"[{r.get('model', '?')}] {r.get('reason', '')}" for r in valid]

        return {
            "sentiment": sentiment,
            "score": avg_score,
            "confidence": avg_confidence,
            "reason": " | ".join(reasons),
            "models_used": [r.get("model", "?") for r in valid],
            "model_scores": {r.get("model", "?"): r.get("score", 50) for r in valid},
        }

    def analyze_single_news(self, news: Dict, index: int = 0, total: int = 0) -> Dict:
        """
        Analyze sentiment for a single news article using dual-model ensemble.

        Both models analyze the same article, results are averaged.

        Args:
            news: News dict with ticker, title, description, published_utc
            index: Current article index (for logging)
            total: Total number of articles (for logging)

        Returns:
            Averaged sentiment result
        """
        prompt = config.SENTIMENT_PROMPT.format(
            ticker=news.get("ticker", "Unknown"),
            title=news.get("title", ""),
            description=news.get("description", ""),
            published_date=news.get("published_utc", "")
        )

        # Call models sequentially to avoid rate limits
        model_results = []
        for model in self.models:
            result = self._analyze_with_single_model(prompt, model)
            model_results.append(result)
            time.sleep(1.0)  # 1s between each model call

        # Average results
        averaged = self._average_model_results(model_results)

        if averaged is None:
            if index and total:
                print(f"  [{index}/{total}] All models failed")
            return self._default_result(news)

        # Attach news metadata
        averaged["title"] = news.get("title", "")
        averaged["source"] = news.get("source", "")
        averaged["published_utc"] = news.get("published_utc", "")

        if index and total:
            scores_str = "  ".join(
                f"{m}:{s}" for m, s in averaged.get("model_scores", {}).items()
            )
            print(f"  [{index}/{total}] {averaged['sentiment']} (avg:{averaged['score']})  [{scores_str}]")

        return averaged

    def _default_result(self, news: Dict) -> Dict:
        """Return a default neutral result on failure"""
        return {
            "sentiment": "neutral",
            "score": 50,
            "confidence": 0,
            "reason": "Analysis failed, using default",
            "models_used": [],
            "model_scores": {},
            "title": news.get("title", ""),
            "source": news.get("source", ""),
            "published_utc": news.get("published_utc", "")
        }

    def analyze_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a batch of news articles.
        Each article is analyzed by all models (ensemble), articles are processed concurrently.

        Args:
            news_list: List of news dicts

        Returns:
            List of averaged sentiment results (in original order)
        """
        total = len(news_list)
        print(f"  Dual-model ensemble: {', '.join(m.split('/')[-1] for m in self.models)}")
        print(f"  Analyzing {total} articles with {self.max_workers} concurrent workers...")

        results = [None] * total

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self.analyze_single_news, news, i + 1, total
                ): i
                for i, news in enumerate(news_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"  [{idx+1}/{total}] Worker error: {e}")
                    results[idx] = self._default_result(news_list[idx])

        return results

    def aggregate_sentiment(self, results: List[Dict]) -> Dict:
        """
        Aggregate sentiment scores from multiple articles into a stock-level score.
        Uses confidence-weighted average.
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

        bullish_count = sum(1 for r in results if r.get("sentiment") == "bullish")
        bearish_count = sum(1 for r in results if r.get("sentiment") == "bearish")
        neutral_count = sum(1 for r in results if r.get("sentiment") == "neutral")

        total_weight = 0
        weighted_score = 0

        for r in results:
            confidence = r.get("confidence", 50)
            score = r.get("score", 50)
            weight = confidence / 100.0 if confidence > 0 else 0.5
            weighted_score += score * weight
            total_weight += weight

        final_score = weighted_score / total_weight if total_weight > 0 else 50

        if final_score >= 60:
            overall_sentiment = "bullish"
        elif final_score <= 40:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"

        avg_confidence = sum(r.get("confidence", 50) for r in results) / len(results)

        return {
            "final_score": round(final_score, 2),
            "sentiment": overall_sentiment,
            "news_count": len(results),
            "avg_confidence": round(avg_confidence, 2),
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "details": results
        }


# Test code
if __name__ == "__main__":
    test_news = [
        {
            "ticker": "AAPL",
            "title": "Apple Reports Record iPhone Sales in Q4",
            "description": "Apple Inc. announced record-breaking iPhone sales, exceeding analyst expectations by 15%.",
            "published_utc": "2026-02-05T10:00:00Z",
            "source": "Reuters"
        },
    ]

    analyzer = SentimentAnalyzer()

    print("=" * 50)
    print("Testing Dual-Model Sentiment Analysis")
    print(f"Models: {', '.join(config.LLM_MODELS)}")
    print("=" * 50)

    results = analyzer.analyze_news_batch(test_news)

    for r in results:
        print(f"\nTitle: {r['title']}")
        print(f"Sentiment: {r['sentiment']} | Avg Score: {r['score']} | Confidence: {r['confidence']}")
        print(f"Model Scores: {r.get('model_scores', {})}")
        print(f"Reason: {r['reason']}")
