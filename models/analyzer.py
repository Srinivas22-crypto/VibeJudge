import json
from typing import Dict, Any, List
import re

class ContentAnalyzer:
    """Analyzes text for sentiment, bias, and tone"""
    
    def __init__(self):
        print("Loading NLP models...")
        # Lazy load heavy libraries
        import spacy
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # Load Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not downloaded
            print("Downloading spacy model...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Load RoBERTa Sentiment Model
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['negative', 'neutral', 'positive']
        
        print("✓ NLP models loaded")
        
        # Load bias lexicon (simplified for MVP)
        self.bias_lexicon = {
            'political_left': ['radical left', 'socialist agenda', 'woke mob', 'comrade'],
            'political_right': ['radical right', 'fascist', 'bootlicker', 'maga'],
            'absolutist': ['always', 'never', 'everyone', 'nobody', 'undeniably'],
            'aggressive': ['stupid', 'idiot', 'shut up', 'disgusting', 'traitor']
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using RoBERTa
        """
        try:
            # Import needed libraries for calculation
            from scipy.special import softmax
            import numpy as np

            # Chunk text if too long (RoBERTa limit 512 tokens)
            # For simplicity, we'll analyze sentence by sentence and average
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
            
            if not sentences:
                return {'positive_pct': 0, 'neutral_pct': 0, 'negative_pct': 0, 'overall_score': 0}
            
            scores = []
            
            # Process in batches or individual sentences (simple approach)
            for sent in sentences[:50]: # Limit to first 50 sentences for speed in MVP
                encoded_input = self.tokenizer(sent, return_tensors='pt', truncation=True, max_length=512)
                output = self.model(**encoded_input)
                score = softmax(output.logits.detach().numpy()[0])
                scores.append(score)
            
            # Average scores
            avg_scores = np.mean(scores, axis=0)
            
            return {
                'negative_pct': round(float(avg_scores[0]) * 100, 1),
                'neutral_pct': round(float(avg_scores[1]) * 100, 1),
                'positive_pct': round(float(avg_scores[2]) * 100, 1),
                'overall_score': round(float(avg_scores[2] - avg_scores[0]), 2) # Range -1 to 1
            }
            
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
            return {}

    def analyze_bias(self, text: str) -> Dict[str, Any]:
        """
        Detect potential bias using lexicon matching
        """
        doc = self.nlp(text.lower())
        flags = []
        score = 0
        
        for category, phrases in self.bias_lexicon.items():
            for phrase in phrases:
                count = text.lower().count(phrase)
                if count > 0:
                    score += count * 10
                    flags.append({
                        'phrase': phrase,
                        'category': category,
                        'count': count
                    })
        
        # Normalize score 0-100 (cap at 100)
        final_score = min(score, 100)
        
        # Determine level
        if final_score < 20: level = "Low"
        elif final_score < 50: level = "Moderate"
        else: level = "High"
        
        return {
            'score': final_score,
            'level': level,
            'flags_count': len(flags),
            'flags': flags
        }

    def analyze_tone(self, text: str) -> Dict[str, Any]:
        """
        Simple keyword-based tone analysis
        """
        # Placeholder for more complex logic
        # For now, derive from sentiment + simple heuristics
        # (Real implementation would use a classifier)
        
        return {
            'dominant_tone': 'Neutral', # To be implemented
            'calm_pct': 50.0,
            'excited_pct': 10.0,
            'aggressive_pct': 0.0
        }

# Singleton
_analyzer_instance = None

def get_analyzer() -> ContentAnalyzer:
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ContentAnalyzer()
    return _analyzer_instance
