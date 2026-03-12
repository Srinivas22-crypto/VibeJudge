"""
Tone Detection Module for VibeJudge
Rule-based + linguistic feature analysis for emotional tone classification
"""

import json
import logging
import re
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import spacy
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToneDetector:
    """
    Detects emotional tone in podcast transcripts using rule-based analysis
    
    Tone Categories:
    - Calm: Measured, factual language
    - Aggressive: Strong language, imperatives, high intensity
    - Persuasive: Modal verbs, rhetorical questions, appeals
    - Anxious: Hedging, uncertainty markers, qualifiers
    - Confident: Assertions, declaratives, minimal hedging
    - Excited: Exclamations, intensifiers, high energy markers
    """
    
    # Linguistic feature dictionaries
    TONE_MARKERS = {
        "aggressive": {
            "keywords": [
                "must", "never", "always", "absolutely", "totally", "completely",
                "obviously", "clearly", "ridiculous", "absurd", "stupid", "idiotic",
                "demand", "insist", "require", "force", "destroy", "attack", "fight"
            ],
            "patterns": [
                r'\b(you|they)\s+(must|should|need to|have to)\b',
                r'\b(never|always|everyone|no one)\b',
                r'!{2,}',  # Multiple exclamation marks
            ],
            "weight": 2.0
        },
        
        "persuasive": {
            "keywords": [
                "should", "could", "would", "might", "perhaps", "consider",
                "imagine", "think about", "what if", "clearly", "obviously",
                "everyone knows", "studies show", "research indicates",
                "believe", "trust me", "honestly", "frankly"
            ],
            "patterns": [
                r'\b(we|you)\s+(should|could|need to)\b',
                r'\b(think about|consider|imagine)\b',
                r'\?$',  # Questions (rhetorical)
            ],
            "weight": 1.5
        },
        
        "anxious": {
            "keywords": [
                "maybe", "possibly", "perhaps", "might", "could be", "I think",
                "I feel", "seems like", "sort of", "kind of", "somewhat",
                "uncertain", "worried", "concerned", "afraid", "nervous",
                "not sure", "don't know", "hopefully"
            ],
            "patterns": [
                r'\b(maybe|perhaps|possibly|might|could be)\b',
                r'\b(I think|I feel|seems|appears)\b',
                r'\b(sort of|kind of|somewhat)\b',
            ],
            "weight": 1.8
        },
        
        "confident": {
            "keywords": [
                "definitely", "certainly", "absolutely", "clearly", "obviously",
                "undoubtedly", "without doubt", "guaranteed", "proven", "fact",
                "I know", "we know", "it is", "this is", "will", "ensure"
            ],
            "patterns": [
                r'\b(I|we)\s+know\b',
                r'\b(it is|this is|that is)\s+\w+\b',
                r'\b(will|shall|must)\b',
            ],
            "weight": 1.7
        },
        
        "excited": {
            "keywords": [
                "amazing", "incredible", "fantastic", "awesome", "wonderful",
                "exciting", "love", "great", "excellent", "brilliant",
                "wow", "omg", "unbelievable", "extraordinary", "phenomenal"
            ],
            "patterns": [
                r'!+',  # Exclamation marks
                r'\b(so|very|really|incredibly|extremely)\s+\w+\b',
                r'\b(love|adore|obsessed)\b',
            ],
            "weight": 1.6
        },
        
        "calm": {
            "keywords": [
                "according to", "data shows", "research indicates", "studies suggest",
                "evidence", "statistics", "analysis", "examine", "consider",
                "however", "nevertheless", "furthermore", "moreover",
                "in conclusion", "to summarize", "essentially"
            ],
            "patterns": [
                r'\b(according to|based on|research|data)\b',
                r'\b(however|nevertheless|furthermore)\b',
            ],
            "weight": 1.0
        }
    }
    
    
    def __init__(self):
        """Initialize tone detector with spaCy NLP"""
        logger.info("Loading spaCy model for tone detection...")
        
        try:
            # Load spaCy with minimal pipeline for efficiency
            self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
            self.nlp.add_pipe('sentencizer')
            logger.info("✓ Tone detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tone detector: {e}")
            raise
    
    
    def analyze_text(self, text: str, segments: List[Dict] = None) -> Dict:
        """
        Analyze tone of transcript text
        
        Args:
            text: Full transcript text
            segments: Whisper segments with timestamps (optional)
        
        Returns:
            Dict with tone analysis results
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result()
        
        logger.info(f"Analyzing tone for text ({len(text)} chars)")
        
        # Segment into sentences
        sentences = self._segment_sentences(text)
        logger.info(f"Split into {len(sentences)} sentences")
        
        # Analyze each sentence
        sentence_results = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:
                result = self._analyze_sentence(sentence, i)
                sentence_results.append(result)
        
        # Calculate overall tone
        overall_tone = self._calculate_overall_tone(sentence_results)
        
        # Generate tone distribution
        tone_distribution = self._calculate_tone_distribution(sentence_results)
        
        # Identify tone examples
        tone_examples = self._extract_tone_examples(sentence_results)
        
        # Generate timeline
        timeline = self._generate_timeline(sentence_results, segments)
        
        return {
            "dominant_tone": overall_tone["label"],
            "dominant_score": overall_tone["score"],
            "confidence": overall_tone["confidence"],
            "tone_distribution": tone_distribution,
            "tone_scores": overall_tone["all_scores"],
            "sentence_count": len(sentence_results),
            "sentences": sentence_results,
            "tone_examples": tone_examples,
            "timeline": timeline,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    
    def _analyze_sentence(self, sentence: str, index: int) -> Dict:
        """
        Analyze tone of a single sentence
        
        Returns:
            Dict with tone label, scores, and features
        """
        # Calculate tone scores for each category
        tone_scores = {}
        feature_matches = {}
        
        for tone_category, markers in self.TONE_MARKERS.items():
            score, matches = self._calculate_tone_score(
                sentence, 
                markers["keywords"],
                markers["patterns"],
                markers["weight"]
            )
            tone_scores[tone_category] = score
            feature_matches[tone_category] = matches
        
        # Add TextBlob polarity and subjectivity
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Adjust scores based on polarity/subjectivity
        if subjectivity > 0.5:
            # High subjectivity boosts persuasive/excited
            tone_scores["persuasive"] *= 1.2
            tone_scores["excited"] *= 1.1
        
        if polarity > 0.5:
            # Very positive polarity boosts excited
            tone_scores["excited"] *= 1.3
        elif polarity < -0.5:
            # Very negative polarity boosts aggressive
            tone_scores["aggressive"] *= 1.2
        
        # Determine dominant tone
        if all(score < 0.1 for score in tone_scores.values()):
            dominant_tone = "calm"
            dominant_score = 0.5
        else:
            dominant_tone = max(tone_scores, key=tone_scores.get)
            dominant_score = tone_scores[dominant_tone]
        
        return {
            "index": index,
            "text": sentence,
            "dominant_tone": dominant_tone,
            "dominant_score": float(dominant_score),
            "tone_scores": {k: float(v) for k, v in tone_scores.items()},
            "polarity": float(polarity),
            "subjectivity": float(subjectivity),
            "features_matched": feature_matches[dominant_tone]
        }
    
    
    def _calculate_tone_score(
        self, 
        sentence: str, 
        keywords: List[str],
        patterns: List[str],
        weight: float
    ) -> Tuple[float, List[str]]:
        """
        Calculate tone score based on keyword and pattern matches
        
        Returns:
            (score, list of matched features)
        """
        sentence_lower = sentence.lower()
        matches = []
        score = 0.0
        
        # Keyword matching
        for keyword in keywords:
            if keyword in sentence_lower:
                matches.append(keyword)
                score += 0.5
        
        # Pattern matching
        for pattern in patterns:
            if re.search(pattern, sentence_lower):
                matches.append(f"pattern:{pattern}")
                score += 0.7
        
        # Apply weight and normalize
        score = score * weight
        score = min(score, 3.0)  # Cap at 3.0
        
        return score, matches
    
    
    def _calculate_overall_tone(self, sentences: List[Dict]) -> Dict:
        """Calculate dominant tone across all sentences"""
        if not sentences:
            return {
                "label": "calm",
                "score": 0.0,
                "confidence": 0.0,
                "all_scores": {}
            }
        
        # Aggregate tone scores
        tone_totals = {
            "calm": 0.0,
            "aggressive": 0.0,
            "persuasive": 0.0,
            "anxious": 0.0,
            "confident": 0.0,
            "excited": 0.0
        }
        
        for sentence in sentences:
            for tone, score in sentence["tone_scores"].items():
                tone_totals[tone] += score
        
        # Normalize by sentence count
        tone_averages = {
            tone: total / len(sentences)
            for tone, total in tone_totals.items()
        }
        
        # Determine dominant tone
        dominant_tone = max(tone_averages, key=tone_averages.get)
        dominant_score = tone_averages[dominant_tone]
        
        # Calculate confidence (difference from second-highest)
        sorted_scores = sorted(tone_averages.values(), reverse=True)
        if len(sorted_scores) > 1:
            confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        else:
            confidence = 1.0
        
        return {
            "label": dominant_tone,
            "score": float(dominant_score),
            "confidence": float(confidence),
            "all_scores": {k: float(v) for k, v in tone_averages.items()}
        }
    
    
    def _calculate_tone_distribution(self, sentences: List[Dict]) -> Dict:
        """Calculate percentage distribution of tones"""
        if not sentences:
            return {}
        
        tone_counts = Counter([s["dominant_tone"] for s in sentences])
        total = len(sentences)
        
        return {
            tone: float(count / total)
            for tone, count in tone_counts.items()
        }
    
    
    def _extract_tone_examples(self, sentences: List[Dict]) -> Dict:
        """Extract representative examples for each tone"""
        tone_examples = {}
        
        # Group sentences by dominant tone
        tone_groups = {}
        for sentence in sentences:
            tone = sentence["dominant_tone"]
            if tone not in tone_groups:
                tone_groups[tone] = []
            tone_groups[tone].append(sentence)
        
        # Get top example for each tone (highest score)
        for tone, group in tone_groups.items():
            sorted_group = sorted(group, key=lambda x: x["dominant_score"], reverse=True)
            top_example = sorted_group[0]
            
            tone_examples[tone] = {
                "text": top_example["text"],
                "score": top_example["dominant_score"],
                "index": top_example["index"]
            }
        
        return tone_examples
    
    
    def _generate_timeline(
        self, 
        sentences: List[Dict], 
        segments: List[Dict] = None
    ) -> List[Dict]:
        """Generate tone timeline in 30-second bins"""
        if not segments:
            return []
        
        # Map sentences to timestamps (simplified approach)
        max_time = max([seg["end"] for seg in segments])
        bin_size = 30
        num_bins = int(np.ceil(max_time / bin_size))
        
        timeline = []
        for i in range(num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size
            
            # Estimate which sentences fall in this bin
            bin_sentences = sentences[
                int(i * len(sentences) / num_bins):
                int((i + 1) * len(sentences) / num_bins)
            ]
            
            if bin_sentences:
                tone_counts = Counter([s["dominant_tone"] for s in bin_sentences])
                dominant_tone = tone_counts.most_common(1)[0][0]
            else:
                dominant_tone = "calm"
            
            timeline.append({
                "time_start": bin_start,
                "time_end": bin_end,
                "time_label": f"{int(bin_start//60):02d}:{int(bin_start%60):02d}",
                "dominant_tone": dominant_tone,
                "sentence_count": len(bin_sentences)
            })
        
        return timeline
    
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "dominant_tone": "calm",
            "dominant_score": 0.0,
            "confidence": 0.0,
            "tone_distribution": {},
            "tone_scores": {},
            "sentence_count": 0,
            "sentences": [],
            "tone_examples": {},
            "timeline": [],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save tone analysis results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Tone results saved to {output_path}")


# Testing function
def test_tone_detector():
    """Test tone detector with sample sentences"""
    detector = ToneDetector()
    
    test_sentences = {
        "calm": "According to recent studies, the data suggests a moderate increase in economic activity.",
        "aggressive": "This is absolutely ridiculous! We must demand immediate action now!",
        "persuasive": "You should really consider the benefits of this approach. Think about what we could achieve.",
        "anxious": "I'm not sure, but maybe we could possibly try that approach if it seems right.",
        "confident": "I know for a fact that this will work. It's clearly the best solution.",
        "excited": "This is amazing! I absolutely love this incredible discovery!"
    }
    
    print("\n" + "="*70)
    print("TONE DETECTOR TEST")
    print("="*70)
    
    for expected_tone, sentence in test_sentences.items():
        result = detector.analyze_text(sentence)
        detected = result["dominant_tone"]
        score = result["dominant_score"]
        
        match = "✓" if detected == expected_tone else "✗"
        print(f"\n{match} Expected: {expected_tone.upper()}, Detected: {detected.upper()}")
        print(f"  Text: {sentence}")
        print(f"  Score: {score:.2f}")
        print(f"  All scores: {result['tone_scores']}")


if __name__ == "__main__":
    test_tone_detector()
