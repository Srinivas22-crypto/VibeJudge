import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import numpy as np

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionPrintAnalyzer:
    """
    Detects emotional authenticity mismatches between
    transcript text and acoustic prosodic features.

    Divergence Categories:
    - Sarcasm      : Positive text + flat/low/slow prosody
    - Irony        : Exaggerated prosody contradicts text meaning
    - Suppression  : Negative text + calm/neutral prosody
    - Authentic    : Text and prosody align → genuine emotion
    """

    # Weights for prosody score components
    PROSODY_WEIGHTS = {
        "pitch_variance": 0.30,   # High variance = more expressive
        "speech_rate":    0.25,   # Fast rate = energetic
        "volume_mean":    0.25,   # High volume = assertive
        "pause_frequency":0.20    # Few pauses = confident
    }

    # Threshold above which divergence is flagged
    DIVERGENCE_THRESHOLD = 0.55

    # Sarcasm-specific prosody thresholds
    SARCASM_THRESHOLDS = {
        "max_pitch_variance_hz": 25.0,   # Very flat pitch
        "max_volume_db":         -22.0,  # Low energy
        "max_speech_rate_syl_s": 3.2     # Slow delivery
    }

    def __init__(self):
        """Initialize EmotionPrint analyzer with lazy model loading"""
        self._librosa = None
        self._sf = None
        self._sentiment_pipeline = None
        logger.info("✓ EmotionPrint™ Analyzer ready (lazy loading)")


    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def analyze_segment(
        self,
        audio_path: str,
        transcript_text: str,
        start_time: float,
        end_time: float,
        segment_id: int = 0
    ) -> Dict:
        """
        Analyze a single audio segment for emotional authenticity.

        Args:
            audio_path   : Path to full podcast audio file
            transcript_text : Transcript text for this segment
            start_time   : Segment start in seconds
            end_time     : Segment end in seconds
            segment_id   : Segment identifier

        Returns:
            Dict with emotion analysis for this segment
        """
        if not transcript_text.strip():
            return self._empty_segment_result(segment_id, start_time, end_time)

        # Step 1: Extract text sentiment score
        text_score = self._get_text_sentiment(transcript_text)

        # Step 2: Extract prosodic features from audio
        prosody = self._extract_prosody(audio_path, start_time, end_time)

        # Step 3: Calculate divergence
        divergence = self._calculate_divergence(text_score, prosody)

        # Step 4: Classify emotional state
        classification = self._classify_emotion(
            text_score, prosody, divergence, transcript_text
        )

        return {
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "timestamp": self._fmt_time(start_time),
            "text": transcript_text,
            "text_sentiment_score": float(text_score),
            "prosody_features": prosody,
            "divergence_score": float(divergence),
            "emotional_state": classification["label"],
            "confidence": float(classification["confidence"]),
            "explanation": classification["explanation"],
            "is_flagged": divergence > self.DIVERGENCE_THRESHOLD
        }


    def analyze_full_transcript(
        self,
        audio_path: str,
        segments: List[Dict],
        sample_every_n: int = 3
    ) -> Dict:
        """
        Analyze entire transcript for EmotionPrint divergences.

        Args:
            audio_path      : Path to podcast audio
            segments        : Whisper segments with start/end/text
            sample_every_n  : Analyze every Nth segment (speed vs coverage)

        Returns:
            Full EmotionPrint analysis report
        """
        if not segments:
            return self._empty_full_result()

        logger.info(f"Running EmotionPrint™ on {len(segments)} segments")

        segment_results = []
        flagged_segments = []
        sarcasm_count = 0
        suppression_count = 0
        irony_count = 0

        for i, seg in enumerate(segments):
            # Sample every N segments for speed
            if i % sample_every_n != 0:
                continue

            result = self.analyze_segment(
                audio_path=audio_path,
                transcript_text=seg.get("text", ""),
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                segment_id=i
            )

            segment_results.append(result)

            if result["is_flagged"]:
                flagged_segments.append(result)

                state = result["emotional_state"]
                if state == "Sarcasm":
                    sarcasm_count += 1
                elif state == "Emotional Suppression":
                    suppression_count += 1
                elif state == "Irony":
                    irony_count += 1

        # Overall authenticity score
        auth_score = self._compute_authenticity_score(segment_results)

        # Key moments
        key_moments = self._identify_key_moments(flagged_segments)

        logger.info(
            f"✓ EmotionPrint™ complete | "
            f"Flagged: {len(flagged_segments)}/{len(segment_results)} | "
            f"Sarcasm: {sarcasm_count}"
        )

        return {
            "total_segments_analyzed": len(segment_results),
            "flagged_segments_count": len(flagged_segments),
            "authenticity_score": auth_score,
            "sarcasm_instances": sarcasm_count,
            "suppression_instances": suppression_count,
            "irony_instances": irony_count,
            "flagged_segments": flagged_segments,
            "all_segments": segment_results,
            "key_moments": key_moments,
            "analysis_timestamp": datetime.now().isoformat(),
            "model": "EmotionPrint™ v1.0"
        }


    # ─────────────────────────────────────────────
    # Text Sentiment
    # ─────────────────────────────────────────────

    def _get_text_sentiment(self, text: str) -> float:
        """
        Get text sentiment score normalized to [-1, +1].

        Uses HuggingFace RoBERTa model.
        Returns:
            float: -1.0 (very negative) to +1.0 (very positive)
        """
        try:
            pipeline = self._load_sentiment_pipeline()
            result = pipeline(text[:512])[0]  # Truncate to 512 tokens

            label = result["label"].lower()
            score = result["score"]

            label_map = {
                "label_0": "negative",
                "label_1": "neutral",
                "label_2": "positive",
                "negative": "negative",
                "neutral": "neutral",
                "positive": "positive"
            }

            label = label_map.get(label, "neutral")

            if label == "positive":
                return float(score)
            elif label == "negative":
                return float(-score)
            else:
                return 0.0

        except Exception as e:
            logger.warning(f"Sentiment pipeline failed: {e}, using TextBlob fallback")
            return self._textblob_sentiment(text)


    def _textblob_sentiment(self, text: str) -> float:
        """TextBlob fallback for sentiment scoring"""
        try:
            from textblob import TextBlob
            return float(TextBlob(text).sentiment.polarity)
        except Exception:
            return 0.0


    def _load_sentiment_pipeline(self):
        """Lazy-load sentiment pipeline"""
        if self._sentiment_pipeline is None:
            from transformers import pipeline
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1  # CPU
            )
        return self._sentiment_pipeline


    # ─────────────────────────────────────────────
    # Prosodic Feature Extraction
    # ─────────────────────────────────────────────

    def _extract_prosody(
        self,
        audio_path: str,
        start_time: float,
        end_time: float
    ) -> Dict:
        """
        Extract 6 prosodic features from audio segment.

        Features:
        1. pitch_mean        - Average fundamental frequency (F0) in Hz
        2. pitch_variance    - F0 standard deviation (expressiveness)
        3. speech_rate       - Estimated syllables/second
        4. volume_mean       - RMS energy in dB
        5. volume_variance   - RMS energy std deviation
        6. pause_frequency   - Count of pauses >300ms per minute

        Returns:
            Dict of prosodic feature values
        """
        try:
            librosa = self._load_librosa()
            import soundfile as sf

            # Load audio segment
            y, sr = librosa.load(
                audio_path,
                sr=16000,
                offset=start_time,
                duration=max(0.5, end_time - start_time),
                mono=True
            )

            if len(y) < 1600:  # Less than 0.1 seconds
                return self._neutral_prosody()

            # ── Feature 1 & 2: Pitch (F0) ──────────────────
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr
            )

            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
            voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]

            pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 150.0
            pitch_variance = float(np.std(voiced_f0)) if len(voiced_f0) > 1 else 0.0

            # ── Feature 3: Speech Rate ───────────────────────
            # Estimate via zero-crossing rate (proxy for syllable rate)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            # Map ZCR to syllables/second (empirical scaling)
            speech_rate = float(zcr_mean * sr / 50)
            speech_rate = np.clip(speech_rate, 0.5, 8.0)

            # ── Feature 4 & 5: Volume (RMS Energy) ──────────
            rms = librosa.feature.rms(y=y)[0]
            rms_mean = float(np.mean(rms))
            rms_variance = float(np.std(rms))

            # Convert to dB
            volume_mean_db = float(20 * np.log10(max(rms_mean, 1e-9)))
            volume_variance_db = float(20 * np.log10(max(rms_variance, 1e-9)))

            # ── Feature 6: Pause Frequency ───────────────────
            # Detect silence intervals
            non_silent = librosa.effects.split(y, top_db=25)
            duration_sec = len(y) / sr

            if len(non_silent) > 1:
                # Count gaps > 0.3 seconds
                pause_count = 0
                for j in range(1, len(non_silent)):
                    gap_samples = non_silent[j][0] - non_silent[j-1][1]
                    gap_sec = gap_samples / sr
                    if gap_sec > 0.3:
                        pause_count += 1

                # Pauses per minute
                pause_freq = pause_count / (duration_sec / 60.0)
            else:
                pause_freq = 0.0

            return {
                "pitch_mean":         round(pitch_mean, 2),
                "pitch_variance":     round(pitch_variance, 2),
                "speech_rate":        round(float(speech_rate), 3),
                "volume_mean_db":     round(volume_mean_db, 2),
                "volume_variance_db": round(volume_variance_db, 2),
                "pause_frequency":    round(pause_freq, 2),
                "duration_sec":       round(duration_sec, 2),
                "extraction_success": True
            }

        except Exception as e:
            logger.warning(f"Prosody extraction failed: {e}")
            return self._neutral_prosody()


    def _neutral_prosody(self) -> Dict:
        """Return neutral prosody when extraction fails"""
        return {
            "pitch_mean":         150.0,
            "pitch_variance":     20.0,
            "speech_rate":        3.5,
            "volume_mean_db":     -20.0,
            "volume_variance_db": -30.0,
            "pause_frequency":    5.0,
            "duration_sec":       0.0,
            "extraction_success": False
        }


    # ─────────────────────────────────────────────
    # Divergence Calculation
    # ─────────────────────────────────────────────

    def _calculate_divergence(
        self,
        text_score: float,
        prosody: Dict
    ) -> float:
        """
        Calculate prosody-semantic divergence score.

        Algorithm:
        1. Normalize each prosodic feature to [0, 1]
        2. Compute weighted prosody score
        3. Divergence = |text_score - prosody_score|

        Returns:
            float: 0.0 (fully aligned) to 1.0 (fully divergent)
        """
        # ── Normalize prosodic features to [0, 1] ──
        # Pitch variance: 0 Hz (flat) → 0.0; 80 Hz (very expressive) → 1.0
        norm_pitch_var = np.clip(prosody["pitch_variance"] / 80.0, 0, 1)

        # Speech rate: 0.5 syl/s (slow) → 0.0; 8.0 syl/s (fast) → 1.0
        norm_speech_rate = np.clip(
            (prosody["speech_rate"] - 0.5) / 7.5, 0, 1
        )

        # Volume: -45 dB (silent) → 0.0; -5 dB (loud) → 1.0
        norm_volume = np.clip(
            (prosody["volume_mean_db"] + 45) / 40.0, 0, 1
        )

        # Pause frequency: 30/min (many pauses) → 0.0; 0/min (no pauses) → 1.0
        norm_pause = np.clip(
            1.0 - (prosody["pause_frequency"] / 30.0), 0, 1
        )

        # ── Weighted prosody score ──
        prosody_score = (
            self.PROSODY_WEIGHTS["pitch_variance"]  * float(norm_pitch_var)  +
            self.PROSODY_WEIGHTS["speech_rate"]     * float(norm_speech_rate) +
            self.PROSODY_WEIGHTS["volume_mean"]     * float(norm_volume)      +
            self.PROSODY_WEIGHTS["pause_frequency"] * float(norm_pause)
        )

        # Map prosody score to [-1, +1] (same scale as text)
        prosody_signed = (prosody_score * 2.0) - 1.0

        # ── Divergence ──
        divergence = abs(text_score - prosody_signed)
        return float(np.clip(divergence, 0, 1))


    # ─────────────────────────────────────────────
    # Emotion Classification
    # ─────────────────────────────────────────────

    def _classify_emotion(
        self,
        text_score: float,
        prosody: Dict,
        divergence: float,
        text: str
    ) -> Dict:
        """
        Classify the emotional state based on text, prosody, and divergence.

        States:
        - Authentic      : Low divergence → text and prosody match
        - Sarcasm        : Positive text + flat/quiet/slow prosody
        - Irony          : Negative text + bright/loud/fast prosody
        - Suppression    : Negative text + neutral/calm prosody
        - Mismatch       : High divergence not matching above patterns
        """
        if divergence <= self.DIVERGENCE_THRESHOLD:
            return {
                "label":       "Authentic",
                "confidence":  1.0 - divergence,
                "explanation": "Text sentiment aligns with vocal delivery."
            }

        pv  = prosody["pitch_variance"]
        vol = prosody["volume_mean_db"]
        sr  = prosody["speech_rate"]

        # ── Sarcasm Rule ──
        # Positive text + flat pitch + low volume + slow speech
        is_sarcasm = (
            text_score > 0.3 and
            pv  < self.SARCASM_THRESHOLDS["max_pitch_variance_hz"] and
            vol < self.SARCASM_THRESHOLDS["max_volume_db"] and
            sr  < self.SARCASM_THRESHOLDS["max_speech_rate_syl_s"]
        )

        if is_sarcasm:
            confidence = self._sarcasm_confidence(text_score, pv, vol, sr)
            return {
                "label": "Sarcasm",
                "confidence": confidence,
                "explanation": (
                    f"Positive text (score: {text_score:.2f}) with flat pitch "
                    f"({pv:.1f} Hz variance), low volume ({vol:.1f} dB), "
                    f"and slow speech ({sr:.1f} syl/s). Likely sarcastic."
                )
            }

        # ── Emotional Suppression Rule ──
        # Negative text + calm/neutral prosody
        is_suppression = (
            text_score < -0.3 and
            pv  < 35.0 and
            vol > -28.0 and
            sr  > 2.5
        )

        if is_suppression:
            confidence = min(0.90, abs(text_score) * 0.8 + divergence * 0.2)
            return {
                "label": "Emotional Suppression",
                "confidence": confidence,
                "explanation": (
                    f"Negative text (score: {text_score:.2f}) delivered with "
                    f"controlled prosody. Speaker may be suppressing emotion."
                )
            }

        # ── Irony Rule ──
        # Negative text + expressive/loud/fast delivery
        is_irony = (
            text_score < -0.2 and
            pv  > 40.0 and
            vol > -18.0
        )

        if is_irony:
            confidence = min(0.85, divergence * 0.7 + 0.15)
            return {
                "label": "Irony",
                "confidence": confidence,
                "explanation": (
                    f"Negative text with unexpectedly expressive vocal delivery. "
                    f"May indicate ironic or exaggerated speech."
                )
            }

        # ── Generic Mismatch ──
        return {
            "label": "Emotional Mismatch",
            "confidence": min(0.75, divergence),
            "explanation": (
                f"Text and vocal delivery diverge (score: {divergence:.2f}). "
                f"Requires manual review."
            )
        }


    def _sarcasm_confidence(
        self,
        text_score: float,
        pitch_var: float,
        volume_db: float,
        speech_rate: float
    ) -> float:
        """
        Calculate sarcasm confidence from feature values.

        Higher text positivity + flatter pitch + lower volume → higher confidence
        """
        # Each feature contributes 0–1 to confidence
        text_contrib   = min(1.0, text_score / 1.0)   * 0.35
        pitch_contrib  = max(0.0, 1.0 - pitch_var / 25.0) * 0.30
        vol_contrib    = max(0.0, 1.0 - (volume_db + 30) / 10.0) * 0.20
        rate_contrib   = max(0.0, 1.0 - speech_rate / 3.2) * 0.15

        confidence = text_contrib + pitch_contrib + vol_contrib + rate_contrib
        return float(np.clip(confidence, 0.50, 0.97))


    # ─────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────

    def _compute_authenticity_score(self, results: List[Dict]) -> float:
        """
        Authenticity Score = % of segments where text and prosody align.
        Range: 0 (fully inauthentic) to 100 (fully authentic).
        """
        if not results:
            return 100.0

        authentic_count = sum(
            1 for r in results
            if r.get("emotional_state") == "Authentic"
        )

        return round(authentic_count / len(results) * 100, 1)


    def _identify_key_moments(self, flagged: List[Dict]) -> Dict:
        """Identify the most significant flagged moments"""
        if not flagged:
            return {
                "highest_sarcasm": None,
                "highest_divergence": None
            }

        sarcasm_instances = [
            f for f in flagged if f["emotional_state"] == "Sarcasm"
        ]

        highest_sarcasm = None
        if sarcasm_instances:
            highest_sarcasm = max(sarcasm_instances, key=lambda x: x["confidence"])

        highest_divergence = max(flagged, key=lambda x: x["divergence_score"])

        return {
            "highest_sarcasm": highest_sarcasm,
            "highest_divergence": highest_divergence
        }


    def _load_librosa(self):
        """Lazy-load librosa"""
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa


    def _fmt_time(self, seconds: float) -> str:
        """Format seconds to MM:SS"""
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"


    def _empty_segment_result(
        self, segment_id: int, start: float, end: float
    ) -> Dict:
        return {
            "segment_id": segment_id,
            "start_time": start,
            "end_time": end,
            "timestamp": self._fmt_time(start),
            "text": "",
            "text_sentiment_score": 0.0,
            "prosody_features": self._neutral_prosody(),
            "divergence_score": 0.0,
            "emotional_state": "Authentic",
            "confidence": 1.0,
            "explanation": "Empty segment.",
            "is_flagged": False
        }


    def _empty_full_result(self) -> Dict:
        return {
            "total_segments_analyzed": 0,
            "flagged_segments_count": 0,
            "authenticity_score": 100.0,
            "sarcasm_instances": 0,
            "suppression_instances": 0,
            "irony_instances": 0,
            "flagged_segments": [],
            "all_segments": [],
            "key_moments": {"highest_sarcasm": None, "highest_divergence": None},
            "analysis_timestamp": datetime.now().isoformat(),
            "model": "EmotionPrint™ v1.0"
        }


    def save_results(self, results: Dict, output_path: str) -> None:
        """Save EmotionPrint results to JSON"""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ EmotionPrint™ results saved to {out}")


