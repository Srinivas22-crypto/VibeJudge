import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyzes sentiment of transcript text using RoBERTa transformer
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = 0 if torch.cuda.is_available() else -1
        
        logger.info(f"Loading sentiment model: {model_name}")
        logger.info(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                max_length=512,
                truncation=True
            )
            
            # Load spaCy for sentence segmentation
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            self.nlp.add_pipe('sentencizer')
            
            logger.info("✓ Sentiment analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            raise
    
    
    def analyze_text(self, text: str, segments: List[Dict] = None) -> Dict:
        """
        Analyze sentiment of transcript text
        
        Args:
            text: Full transcript text
            segments: Whisper segments with timestamps (optional)
        
        Returns:
            Dict with sentiment analysis results
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result()
        
        logger.info(f"Analyzing sentiment for text ({len(text)} chars)")
        
        # Sentence segmentation
        sentences = self._segment_sentences(text)
        logger.info(f"Split into {len(sentences)} sentences")
        
        # Analyze each sentence
        sentence_results = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Skip very short sentences
                result = self._analyze_sentence(sentence, i)
                sentence_results.append(result)
        
        # Calculate overall statistics
        overall = self._calculate_overall_sentiment(sentence_results)
        
        # Generate timeline data (30-second bins)
        timeline = self._generate_timeline(sentence_results, segments)
        
        # Identify key moments
        key_moments = self._identify_key_moments(sentence_results)
        
        return {
            "overall_sentiment": overall["label"],
            "overall_score": overall["score"],
            "confidence": overall["confidence"],
            "sentence_count": len(sentence_results),
            "positive_ratio": overall["positive_ratio"],
            "negative_ratio": overall["negative_ratio"],
            "neutral_ratio": overall["neutral_ratio"],
            "sentences": sentence_results,
            "timeline": timeline,
            "key_moments": key_moments,
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": self.model_name
        }
    
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    
    def _analyze_sentence(self, sentence: str, index: int) -> Dict:
        """
        Analyze single sentence sentiment
        
        Returns:
            Dict with label, score, confidence
        """
        try:
            result = self.sentiment_pipeline(sentence)[0]
            
            # Map RoBERTa labels to standard labels
            label_map = {
                "LABEL_0": "negative",
                "LABEL_1": "neutral",
                "LABEL_2": "positive",
                "negative": "negative",
                "neutral": "neutral",
                "positive": "positive"
            }
            
            label = label_map.get(result["label"], result["label"]).lower()
            score = result["score"]
            
            # Convert to -1 to +1 scale
            if label == "positive":
                normalized_score = score
            elif label == "negative":
                normalized_score = -score
            else:
                normalized_score = 0.0
            
            return {
                "index": index,
                "text": sentence,
                "label": label,
                "score": normalized_score,
                "confidence": score
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze sentence {index}: {e}")
            return {
                "index": index,
                "text": sentence,
                "label": "neutral",
                "score": 0.0,
                "confidence": 0.0
            }
    
    
    def _calculate_overall_sentiment(self, sentences: List[Dict]) -> Dict:
        """Calculate aggregate sentiment statistics"""
        if not sentences:
            return {
                "label": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "neutral_ratio": 0.0
            }
        
        scores = [s["score"] for s in sentences]
        labels = [s["label"] for s in sentences]
        confidences = [s["confidence"] for s in sentences]
        
        # Overall score (weighted average)
        overall_score = np.mean(scores)
        
        # Determine overall label
        if overall_score > 0.1:
            overall_label = "positive"
        elif overall_score < -0.1:
            overall_label = "negative"
        else:
            overall_label = "neutral"
        
        # Calculate ratios
        total = len(labels)
        positive_ratio = labels.count("positive") / total
        negative_ratio = labels.count("negative") / total
        neutral_ratio = labels.count("neutral") / total
        
        return {
            "label": overall_label,
            "score": float(overall_score),
            "confidence": float(np.mean(confidences)),
            "positive_ratio": float(positive_ratio),
            "negative_ratio": float(negative_ratio),
            "neutral_ratio": float(neutral_ratio)
        }
    
    
    def _generate_timeline(
        self, 
        sentences: List[Dict], 
        segments: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Generate sentiment timeline in 30-second bins
        
        Args:
            sentences: Analyzed sentences
            segments: Whisper segments with timestamps
        
        Returns:
            List of timeline data points
        """
        if not segments:
            # If no timestamps, distribute evenly
            return self._timeline_without_timestamps(sentences)
        
        # Map sentences to timestamps using Whisper segments
        sentence_timestamps = self._map_sentences_to_timestamps(sentences, segments)
        
        # Determine total duration
        max_time = max([seg["end"] for seg in segments])
        
        # Create 30-second bins
        bin_size = 30  # seconds
        num_bins = int(np.ceil(max_time / bin_size))
        
        timeline = []
        for i in range(num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size
            
            # Get sentences in this bin
            bin_sentences = [
                s for s in sentence_timestamps 
                if bin_start <= s["timestamp"] < bin_end
            ]
            
            if bin_sentences:
                avg_score = np.mean([s["score"] for s in bin_sentences])
                dominant_label = max(
                    set([s["label"] for s in bin_sentences]),
                    key=[s["label"] for s in bin_sentences].count
                )
            else:
                avg_score = 0.0
                dominant_label = "neutral"
            
            timeline.append({
                "time_start": bin_start,
                "time_end": bin_end,
                "time_label": f"{int(bin_start//60):02d}:{int(bin_start%60):02d}",
                "avg_sentiment": float(avg_score),
                "dominant_label": dominant_label,
                "sentence_count": len(bin_sentences)
            })
        
        return timeline
    
    
    def _timeline_without_timestamps(self, sentences: List[Dict]) -> List[Dict]:
        """Generate timeline when timestamps are not available"""
        # Divide sentences into 10 equal bins
        num_bins = min(10, len(sentences))
        bin_size = len(sentences) // num_bins
        
        timeline = []
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sentences)
            
            bin_sentences = sentences[start_idx:end_idx]
            avg_score = np.mean([s["score"] for s in bin_sentences])
            
            timeline.append({
                "segment": i + 1,
                "avg_sentiment": float(avg_score),
                "sentence_count": len(bin_sentences)
            })
        
        return timeline
    
    
    def _map_sentences_to_timestamps(
        self, 
        sentences: List[Dict], 
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Map sentences to approximate timestamps using Whisper segments
        
        Strategy: Match sentence text to segment text
        """
        sentence_timestamps = []
        
        for sentence in sentences:
            sentence_text = sentence["text"].lower().strip()
            
            # Find best matching segment
            best_match = None
            best_overlap = 0
            
            for segment in segments:
                segment_text = segment["text"].lower().strip()
                
                # Calculate word overlap
                sentence_words = set(sentence_text.split())
                segment_words = set(segment_text.split())
                overlap = len(sentence_words & segment_words)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = segment
            
            if best_match:
                sentence_timestamps.append({
                    **sentence,
                    "timestamp": best_match["start"]
                })
            else:
                # Default to index-based estimation
                sentence_timestamps.append({
                    **sentence,
                    "timestamp": sentence["index"] * 5  # Assume 5 sec/sentence
                })
        
        return sentence_timestamps
    
    
    def _identify_key_moments(self, sentences: List[Dict]) -> Dict:
        """Identify most positive and most negative moments"""
        if not sentences:
            return {
                "most_positive": None,
                "most_negative": None
            }
        
        # Sort by score
        sorted_sentences = sorted(sentences, key=lambda x: x["score"])
        
        return {
            "most_negative": {
                "text": sorted_sentences[0]["text"],
                "score": sorted_sentences[0]["score"],
                "index": sorted_sentences[0]["index"]
            },
            "most_positive": {
                "text": sorted_sentences[-1]["text"],
                "score": sorted_sentences[-1]["score"],
                "index": sorted_sentences[-1]["index"]
            }
        }
    
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "overall_sentiment": "neutral",
            "overall_score": 0.0,
            "confidence": 0.0,
            "sentence_count": 0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 0.0,
            "sentences": [],
            "timeline": [],
            "key_moments": {"most_positive": None, "most_negative": None},
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": self.model_name
        }
    
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save sentiment analysis results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Sentiment results saved to {output_path}")
    
    
    def load_results(self, input_path: str) -> Dict:
        """Load sentiment analysis results from JSON file"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# Testing function
def test_sentiment_analyzer():
    """Test sentiment analyzer with sample texts"""
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "This is an amazing podcast! I absolutely love the insights shared here.",
        "This is terrible. I completely disagree with these points.",
        "Today we're discussing the economic policies. Let's look at the data.",
        "Oh great, another delay. That's exactly what we needed."  # Sarcasm test
    ]
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYZER TEST")
    print("="*60)
    
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text(text)
        print(f"\n{i}. Text: {text}")
        print(f"   Sentiment: {result['overall_sentiment'].upper()}")
        print(f"   Score: {result['overall_score']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")


if __name__ == "__main__":
    test_sentiment_analyzer()