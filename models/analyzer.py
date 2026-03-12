
from typing import Dict, Any, List

class VibeAnalyzer:
    def __init__(self):
        # Basic lexicons for MVP
        self.positive_words = {"good", "great", "excellent", "best", "amazing", "love", "like", "wonderful", "fantastic", "happy"}
        self.negative_words = {"bad", "worst", "terrible", "hate", "awful", "horrible", "sad", "angry", "poor", "wrong"}
        
        self.bias_words = {
            "obviously": "opinionated",
            "clearly": "opinionated",
            "everyone knows": "generalization",
            "always": "absolute",
            "never": "absolute",
            "radical": "political",
            "extremist": "political",
            "agenda": "conspiracy",
            "mainstream media": "political warning"
        }
        
        self.tone_markers = {
            "aggressive": ["stupid", "idiot", "fight", "war", "attack", "destroy"],
            "excited": ["wow", "!", "amazing", "unbelievable", "huge"],
            "calm": ["peace", "calm", "relax", "listen", "understand", "consider"]
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of the text.
        """
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in self.positive_words)
        neg_count = sum(1 for w in words if w in self.negative_words)
        total = max(1, len(words))
        
        # Calculate percentages
        positive_pct = (pos_count / total) * 100 * 5
        positive_pct = min(100, max(0, positive_pct))
        
        negative_pct = (neg_count / total) * 100 * 5
        negative_pct = min(100, max(0, negative_pct))
        
        neutral_pct = max(0, 100 - positive_pct - negative_pct)
        
        score = (pos_count - neg_count) / max(1, pos_count + neg_count)
        
        overall = "Neutral"
        if score > 0.1: overall = "Positive"
        elif score < -0.1: overall = "Negative"
            
        return {
            'overall_score': overall,
            'positive_pct': round(positive_pct, 1),
            'negative_pct': round(negative_pct, 1),
            'neutral_pct': round(neutral_pct, 1),
            'raw_score': score
        }

    def analyze_bias(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for bias indicators.
        """
        flags = []
        lower_text = text.lower()
        
        for phrase, category in self.bias_words.items():
            if phrase in lower_text:
                # Find context (mock)
                idx = lower_text.find(phrase)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(phrase) + 20)
                context = text[start:end]
                
                flags.append({
                    'phrase': phrase,
                    'category': category,
                    'severity': 'medium',
                    'context': f"...{context}...",
                    'sentence': context,
                    'timestamp': '00:00',
                    'timestamp_seconds': 0.0
                })
        
        score = len(flags) * 10
        level = "Low"
        if score > 30: level = "Moderate" 
        if score > 60: level = "High"
            
        return {
            'level': level,
            'score': min(100, score),
            'flags': flags,
            'flags_count': len(flags)
        }

    def analyze_tone(self, text: str) -> Dict[str, Any]:
        """
        Analyze dominant tone and percentages.
        """
        lower_text = text.lower()
        scores = {tone: 0 for tone in self.tone_markers}
        total_hits = 0
        
        for tone, markers in self.tone_markers.items():
            for marker in markers:
                if marker in lower_text:
                    scores[tone] += 1
                    total_hits += 1
                    
        dominant = max(scores, key=scores.get)
        if scores[dominant] == 0:
            dominant = "neutral"
            
        # Calculate percentages
        metrics = {
            'dominant_tone': dominant.capitalize()
        }
        
        # Add basic defaults for other expected keys if total_hits is 0
        basic_tones = ['calm', 'aggressive', 'excited', 'persuasive', 'anxious', 'confident']
        
        if total_hits > 0:
            for tone in basic_tones:
                if tone in scores:
                    metrics[f'{tone}_pct'] = round((scores[tone] / total_hits) * 100, 1)
                else:
                    metrics[f'{tone}_pct'] = 0
        else:
            for tone in basic_tones:
                metrics[f'{tone}_pct'] = 0
            if dominant == 'neutral':
               metrics['calm_pct'] = 100 # Default to calm if neutral
               
        return metrics

_analyzer_instance = None

def get_analyzer():
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = VibeAnalyzer()
    return _analyzer_instance
