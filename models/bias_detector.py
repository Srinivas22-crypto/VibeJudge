import json
import logging
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import spacy
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Detects linguistic bias in podcast transcripts
    
    Features:
    - Keyword matching across 5 bias categories
    - Named Entity Recognition (NER) for context
    - Temporal context extraction (±30s audio, ±2 sentences text)
    - Bias intensity scoring (0-100 scale)
    """
    
    def __init__(self, dictionary_path: str = "data/bias_keywords.json"):
        """
        Initialize bias detector
        
        Args:
            dictionary_path: Path to bias keyword dictionary JSON
        """
        logger.info("Initializing Bias Detector...")
        
        # Load bias dictionary
        self.bias_dict = self._load_bias_dictionary(dictionary_path)
        logger.info(f"Loaded {len(self.bias_dict)} bias categories")
        
        # Load spaCy for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✓ spaCy NER model loaded")
        except OSError:
            logger.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
        
        logger.info("✓ Bias Detector initialized")
    
    
    def _load_bias_dictionary(self, path: str) -> Dict:
        """Load bias keyword dictionary from JSON"""
        dict_path = Path(path)
        
        if not dict_path.exists():
            logger.warning(f"Dictionary not found at {path}, creating default")
            self._create_default_dictionary(dict_path)
        
        with open(dict_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    
    def _create_default_dictionary(self, path: Path):
        """Create default bias dictionary if not exists"""
        default_dict = {
            "political_left": {
                "keywords": ["socialist agenda", "left-wing extremists", "liberal elite"],
                "severity": "HIGH"
            },
            "political_right": {
                "keywords": ["right-wing fanatics", "conservative conspiracy", "alt-right"],
                "severity": "HIGH"
            },
            "gender_bias": {
                "keywords": ["emotional woman", "bossy", "hysterical"],
                "severity": "MEDIUM"
            },
            "loaded_language": {
                "keywords": ["terrorist", "thug", "regime"],
                "severity": "HIGH"
            },
            "weasel_words": {
                "keywords": ["some people say", "many believe", "critics argue"],
                "severity": "MEDIUM"
            }
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default_dict, f, indent=2)
        
        logger.info(f"Created default dictionary at {path}")
    
    
    def analyze_text(
        self, 
        text: str, 
        segments: Optional[List[Dict]] = None,
        audio_path: Optional[str] = None
    ) -> Dict:
        """
        Analyze text for bias with temporal context
        
        Args:
            text: Full transcript text
            segments: Whisper segments with timestamps
            audio_path: Path to original audio file (for context extraction)
        
        Returns:
            Dict with bias analysis results
        """
        if not text or len(text.strip()) == 0:
            return self._empty_result()
        
        logger.info(f"Analyzing bias in text ({len(text)} chars)")
        
        # Segment into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        logger.info(f"Split into {len(sentences)} sentences")
        
        # Detect bias instances
        bias_flags = []
        
        for category, data in self.bias_dict.items():
            keywords = data["keywords"]
            severity = data["severity"]
            
            for keyword in keywords:
                matches = self._find_keyword_matches(
                    text, 
                    keyword, 
                    sentences,
                    category,
                    severity
                )
                bias_flags.extend(matches)
        
        logger.info(f"Found {len(bias_flags)} potential bias instances")
        
        # Enhance with NER context
        bias_flags = self._add_ner_context(bias_flags, doc)
        
        # Add timestamps if available
        if segments:
            bias_flags = self._add_timestamps(bias_flags, segments)
        
        # Extract audio context if available
        if audio_path and segments:
            bias_flags = self._extract_audio_context(bias_flags, audio_path)
        
        # Calculate overall bias score
        overall_score = self._calculate_bias_score(bias_flags)
        
        # Generate timeline
        timeline = self._generate_timeline(bias_flags, segments)
        
        # Category distribution
        category_dist = self._calculate_category_distribution(bias_flags)
        
        return {
            "bias_flags_count": len(bias_flags),
            "overall_bias_score": overall_score["score"],
            "bias_level": overall_score["level"],
            "category_distribution": category_dist,
            "bias_flags": bias_flags,
            "timeline": timeline,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    
    def _find_keyword_matches(
        self,
        text: str,
        keyword: str,
        sentences: List[str],
        category: str,
        severity: str
    ) -> List[Dict]:
        """
        Find all matches of keyword in text
        
        Returns:
            List of match dictionaries
        """
        matches = []
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Find all occurrences
        start = 0
        while True:
            pos = text_lower.find(keyword_lower, start)
            if pos == -1:
                break
            
            # Find containing sentence
            containing_sentence = self._find_containing_sentence(pos, sentences, text)
            
            # Extract text context (±2 sentences)
            text_context = self._extract_text_context(containing_sentence, sentences)
            
            match = {
                "keyword": keyword,
                "category": category,
                "severity": severity,
                "position": pos,
                "sentence": containing_sentence,
                "text_context": text_context,
                "entities": [],  # Will be filled by NER
                "timestamp": None,  # Will be filled if segments available
                "audio_context_path": None  # Will be filled if audio available
            }
            
            matches.append(match)
            start = pos + len(keyword)
        
        return matches
    
    
    def _find_containing_sentence(
        self, 
        position: int, 
        sentences: List[str],
        full_text: str
    ) -> str:
        """Find which sentence contains the character position"""
        current_pos = 0
        
        for sentence in sentences:
            sentence_start = full_text.find(sentence, current_pos)
            sentence_end = sentence_start + len(sentence)
            
            if sentence_start <= position < sentence_end:
                return sentence
            
            current_pos = sentence_end
        
        return sentences[0] if sentences else ""
    
    
    def _extract_text_context(
        self, 
        target_sentence: str, 
        all_sentences: List[str]
    ) -> str:
        """
        Extract ±2 sentences around target
        
        Returns:
            Context string with surrounding sentences
        """
        try:
            idx = all_sentences.index(target_sentence)
        except ValueError:
            return target_sentence
        
        start_idx = max(0, idx - 2)
        end_idx = min(len(all_sentences), idx + 3)
        
        context_sentences = all_sentences[start_idx:end_idx]
        return " ".join(context_sentences)
    
    
    def _add_ner_context(self, bias_flags: List[Dict], doc) -> List[Dict]:
        """
        Add named entities found in context to each bias flag
        
        Args:
            bias_flags: List of bias flag dictionaries
            doc: spaCy Doc object
        
        Returns:
            Enhanced bias_flags with entity information
        """
        for flag in bias_flags:
            context_text = flag["text_context"]
            
            # Process context with spaCy
            context_doc = self.nlp(context_text)
            
            # Extract entities
            entities = []
            for ent in context_doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT"]:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            
            flag["entities"] = entities
        
        return bias_flags
    
    
    def _add_timestamps(
        self, 
        bias_flags: List[Dict], 
        segments: List[Dict]
    ) -> List[Dict]:
        """
        Add timestamps to bias flags by matching with Whisper segments
        
        Strategy: Find segment containing the flagged sentence
        """
        for flag in bias_flags:
            flagged_sentence = flag["sentence"].lower()
            
            # Find best matching segment
            best_match = None
            best_overlap = 0
            
            for segment in segments:
                segment_text = segment["text"].lower()
                
                # Calculate word overlap
                flag_words = set(flagged_sentence.split())
                segment_words = set(segment_text.split())
                overlap = len(flag_words & segment_words)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = segment
            
            if best_match:
                flag["timestamp"] = best_match["start"]
                flag["timestamp_formatted"] = self._format_timestamp(best_match["start"])
        
        return bias_flags
    
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    
    def _extract_audio_context(
        self, 
        bias_flags: List[Dict], 
        audio_path: str
    ) -> List[Dict]:
        """
        Extract ±30 second audio clips around each bias flag
        
        Args:
            bias_flags: List with timestamps
            audio_path: Path to original audio file
        
        Returns:
            Enhanced bias_flags with audio context paths
        """
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            logger.warning(f"Could not load audio for context extraction: {e}")
            return bias_flags
        
        # Create output directory
        context_dir = Path("data/bias_contexts")
        context_dir.mkdir(parents=True, exist_ok=True)
        
        for i, flag in enumerate(bias_flags):
            if flag["timestamp"] is None:
                continue
            
            timestamp_ms = int(flag["timestamp"] * 1000)
            
            # Extract ±30 seconds
            start_ms = max(0, timestamp_ms - 30000)
            end_ms = min(len(audio), timestamp_ms + 30000)
            
            context_clip = audio[start_ms:end_ms]
            
            # Save clip
            clip_filename = f"bias_context_{i}_{int(flag['timestamp'])}.wav"
            clip_path = context_dir / clip_filename
            
            context_clip.export(str(clip_path), format="wav")
            
            flag["audio_context_path"] = str(clip_path)
            flag["audio_context_duration"] = len(context_clip) / 1000.0
        
        logger.info(f"Extracted {len([f for f in bias_flags if f['audio_context_path']])} audio contexts")
        
        return bias_flags
    
    
    def _calculate_bias_score(self, bias_flags: List[Dict]) -> Dict:
        """
        Calculate overall bias score (0-100 scale)
        
        Formula:
        - Each HIGH severity flag: +10 points
        - Each MEDIUM severity flag: +5 points
        - Normalized to 0-100 scale
        """
        if not bias_flags:
            return {"score": 0, "level": "Low"}
        
        total_score = 0
        
        for flag in bias_flags:
            if flag["severity"] == "HIGH":
                total_score += 10
            elif flag["severity"] == "MEDIUM":
                total_score += 5
            else:
                total_score += 2
        
        # Normalize to 0-100 (assume 10 flags = 100 score as baseline)
        normalized_score = min(100, (total_score / 100) * 100)
        
        # Determine level
        if normalized_score < 20:
            level = "Low"
        elif normalized_score < 50:
            level = "Moderate"
        else:
            level = "High"
        
        return {
            "score": float(normalized_score),
            "level": level
        }
    
    
    def _generate_timeline(
        self, 
        bias_flags: List[Dict], 
        segments: Optional[List[Dict]]
    ) -> List[Dict]:
        """
        Generate bias intensity timeline in 30-second bins
        
        Returns:
            List of timeline data points
        """
        if not segments or not bias_flags:
            return []
        
        # Determine duration
        max_time = max([seg["end"] for seg in segments])
        
        # Create 30-second bins
        bin_size = 30
        num_bins = int(np.ceil(max_time / bin_size))
        
        timeline = []
        
        for i in range(num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size
            
            # Count bias flags in this bin
            flags_in_bin = [
                f for f in bias_flags 
                if f["timestamp"] and bin_start <= f["timestamp"] < bin_end
            ]
            
            # Calculate bin intensity
            intensity = len(flags_in_bin)
            
            timeline.append({
                "time_start": bin_start,
                "time_end": bin_end,
                "time_label": self._format_timestamp(bin_start),
                "bias_count": intensity,
                "flags": [
                    {
                        "keyword": f["keyword"],
                        "category": f["category"],
                        "timestamp": f["timestamp_formatted"]
                    }
                    for f in flags_in_bin
                ]
            })
        
        return timeline
    
    
    def _calculate_category_distribution(self, bias_flags: List[Dict]) -> Dict:
        """Calculate percentage distribution across categories"""
        if not bias_flags:
            return {}
        
        category_counts = defaultdict(int)
        
        for flag in bias_flags:
            category_counts[flag["category"]] += 1
        
        total = len(bias_flags)
        
        return {
            category: count / total
            for category, count in category_counts.items()
        }
    
    
    def _empty_result(self) -> Dict:
        """Return empty result structure"""
        return {
            "bias_flags_count": 0,
            "overall_bias_score": 0.0,
            "bias_level": "Low",
            "category_distribution": {},
            "bias_flags": [],
            "timeline": [],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save bias analysis results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Bias results saved to {output_path}")
