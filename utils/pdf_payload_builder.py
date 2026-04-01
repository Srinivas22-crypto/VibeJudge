# utils/pdf_payload_builder.py

def safe_get(d, *keys, default=None):
    """
    Safely get nested keys from dict.
    """
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def compute_word_count(transcript_text: str) -> int:
    if not transcript_text:
        return 0
    return len(transcript_text.split())


def compute_duration_min(segments: list) -> float:
    if not segments:
        return 0.0
    last_end = max(seg.get("end", 0) for seg in segments if isinstance(seg, dict))
    return round(last_end / 60.0, 2)


def build_pdf_payload(results: dict, uploaded_filename: str = "Unknown"):
    """
    Convert your real pipeline results into the exact structure
    expected by generate_analysis_pdf().
    """

    # ---- Transcript extraction ----
    transcript_block = results.get("transcription") or results.get("transcript") or {}
    transcript_text = (
        transcript_block.get("text")
        or results.get("transcript_text")
        or ""
    )

    transcript_segments = transcript_block.get("segments", [])

    word_count = (
        transcript_block.get("word_count")
        or compute_word_count(transcript_text)
    )

    duration_min = (
        results.get("duration_min")
        or compute_duration_min(transcript_segments)
    )

    # ---- Sentiment ----
    sentiment_block = results.get("sentiment", {})
    sentiment_summary = sentiment_block.get("summary", sentiment_block)

    # ---- Tone ----
    tone_block = results.get("tone", {})
    tone_summary = tone_block.get("summary", tone_block)

    # ---- Bias ----
    bias_block = results.get("bias", {})
    bias_summary = bias_block.get("summary", bias_block)

    # ---- EmotionPrint ----
    emotion_block = results.get("emotionprint", {}) or results.get("emotion_print", {})
    emotion_summary = emotion_block.get("summary", emotion_block)

    # Normalize authenticity field
    authenticity_value = (
        emotion_summary.get("authenticity_pct")
        or emotion_summary.get("authenticity")
        or "N/A"
    )

    payload = {
        "podcast_name": results.get("podcast_name", uploaded_filename),
        "duration_min": duration_min,
        "transcription": {
            "word_count": word_count,
            "segments": transcript_segments
        },
        "sentiment": {
            "summary": {
                "overall_label": sentiment_summary.get("overall_label", "N/A"),
                "overall_score": sentiment_summary.get("overall_score", "N/A"),
                "confidence": sentiment_summary.get("confidence", "N/A"),
                "sentence_count": sentiment_summary.get("sentence_count", "N/A"),
                "distribution": sentiment_summary.get("distribution", {}),
                "most_positive": sentiment_summary.get("most_positive", []),
                "most_negative": sentiment_summary.get("most_negative", [])
            }
        },
        "tone": {
            "summary": {
                "dominant_tone": tone_summary.get("dominant_tone", "N/A"),
                "tone_score": tone_summary.get("tone_score", "N/A"),
                "confidence": tone_summary.get("confidence", "N/A"),
                "distribution": tone_summary.get("distribution", {})
            }
        },
        "bias": {
            "summary": {
                "bias_score": bias_summary.get("bias_score", "N/A"),
                "bias_level": bias_summary.get("bias_level", "N/A"),
                "total_flags": bias_summary.get("total_flags", 0)
            },
            "category_distribution": bias_block.get("category_distribution", {}),
            "flagged_instances": bias_block.get("flagged_instances", [])
        },
        "emotionprint": {
            "summary": {
                "authenticity_pct": authenticity_value,
                "mismatch_count": emotion_summary.get("mismatch_count", 0),
                "sarcasm_count": emotion_summary.get("sarcasm_count", 0),
                "suppression_count": emotion_summary.get("suppression_count", 0),
                "irony_count": emotion_summary.get("irony_count", 0)
            },
            "flagged_moments": emotion_block.get("flagged_moments", [])
        },
        "transcript_text": transcript_text
    }

    return payload
