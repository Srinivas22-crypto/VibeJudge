import json
import uuid
import logging
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────
# Path Resolution Fix (Streamlit)
# ─────────────────────────────────────────────────────────
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import plotly.graph_objects as go

from models.sentiment_analyzer import SentimentAnalyzer
from models.tone_detector import ToneDetector
from models.bias_detector import BiasDetector
from models.emotion_print import EmotionPrintAnalyzer
from database.db_manager import DatabaseManager
from utils.visualizations import (
    create_sentiment_timeline,
    create_sentiment_distribution_pie,
    create_combined_dashboard,
    create_bias_timeline,
    create_bias_category_chart,
    create_emotionprint_timeline,
    create_emotionprint_summary_chart,
    generate_color_coded_transcript
)
from utils.pdf_generator import generate_analysis_pdf
from config import settings as config
from utils.audio_preprocessing import convert_to_wav_16k_mono
from utils.audio_chunking import split_audio_into_chunks
from utils.parallel_transcription import transcribe_chunks_parallel
from utils.transcript_merger import merge_chunk_transcripts, save_merged_transcript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeJudge — Podcast Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "analysis_done": False,
        "analysis_results": None,
        "pdf_payload": None,
        "pdf_bytes": None,
        "uploaded_filename": None,
        "uploaded_file_size": None,
        "charts": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 6px;
        margin-top: 24px;
    }
    .badge-high   { color: #c0392b; font-weight: bold; }
    .badge-medium { color: #e67e22; font-weight: bold; }
    .badge-low    { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🎙️ VibeJudge")
        st.markdown("*Podcast Intelligence Platform*")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Analyze", "📊 Dashboard", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("**Analysis Modules**")
        st.markdown("✅ Transcription (Whisper)")
        st.markdown("✅ Sentiment (RoBERTa)")
        st.markdown("✅ Tone Detection")
        st.markdown("✅ Bias Detection")
        st.markdown("✅ EmotionPrint™")
        st.markdown("---")
        st.caption("VibeJudge v1.3 · Week 3")

    return page

# ─────────────────────────────────────────────────────────
# File Validation
# ─────────────────────────────────────────────────────────
def validate_uploaded_file(uploaded_file) -> tuple:
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    ext = Path(uploaded_file.name).suffix.lower()

    if ext not in allowed:
        return False, f"Unsupported format '{ext}'. Use: {', '.join(sorted(allowed))}"

    if uploaded_file.size > 100 * 1024 * 1024:
        size_mb = uploaded_file.size / 1024 / 1024
        return False, f"File too large ({size_mb:.1f} MB). Maximum is 100 MB."

    return True, ""

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def safe_pct(value):
    if value is None:
        return 0.0
    try:
        value = float(value)
        return round(value * 100, 2) if value <= 1 else round(value, 2)
    except Exception:
        return 0.0

def create_tone_distribution_chart(tone: dict):
    if not tone.get("tone_distribution"):
        return None

    labels = list(tone["tone_distribution"].keys())
    values = [tone["tone_distribution"][label] * 100 for label in labels]

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=[
            "#3498db", "#e74c3c", "#f39c12",
            "#9b59b6", "#2ecc71", "#1abc9c"
        ][:len(labels)]
    )])
    fig.update_layout(
        title="Tone Distribution (%)",
        xaxis_title="Tone",
        yaxis_title="Percentage",
        template="plotly_white",
        height=380
    )
    return fig

def create_pdf_charts(results: dict) -> dict:
    charts = {}
    sentiment = results.get("sentiment", {})
    tone = results.get("tone", {})
    bias = results.get("bias", {})
    ep = results.get("emotionprint", {})

    try:
        if sentiment.get("timeline"):
            charts["sentiment_timeline"] = create_sentiment_timeline(sentiment["timeline"])
    except Exception as e:
        logger.warning(f"Could not create sentiment timeline chart: {e}")

    try:
        charts["sentiment_distribution"] = create_sentiment_distribution_pie(sentiment)
    except Exception as e:
        logger.warning(f"Could not create sentiment distribution chart: {e}")

    try:
        tone_fig = create_tone_distribution_chart(tone)
        if tone_fig is not None:
            charts["tone_distribution"] = tone_fig
    except Exception as e:
        logger.warning(f"Could not create tone chart: {e}")

    try:
        if bias.get("timeline"):
            charts["bias_timeline"] = create_bias_timeline(bias["timeline"])
    except Exception as e:
        logger.warning(f"Could not create bias timeline chart: {e}")

    try:
        if bias.get("category_distribution"):
            charts["bias_distribution"] = create_bias_category_chart(bias["category_distribution"])
    except Exception as e:
        logger.warning(f"Could not create bias category chart: {e}")

    try:
        charts["emotion_timeline"] = create_emotionprint_timeline(ep)
    except Exception as e:
        logger.warning(f"Could not create emotionprint timeline chart: {e}")

    try:
        charts["emotion_summary"] = create_emotionprint_summary_chart(ep)
    except Exception as e:
        logger.warning(f"Could not create emotionprint summary chart: {e}")

    return charts

def build_pdf_payload(results: dict) -> dict:
    """
    Convert your actual result structure into the structure
    expected by generate_analysis_pdf().
    """
    transcript = results.get("transcript", {})
    sentiment = results.get("sentiment", {})
    tone = results.get("tone", {})
    bias = results.get("bias", {})
    ep = results.get("emotionprint", {})

    transcript_text = transcript.get("text", "")
    transcript_segments = transcript.get("segments", [])

    word_count = transcript.get("word_count", 0)
    duration_sec = transcript.get("duration", 0)
    duration_min = round(duration_sec / 60, 2) if duration_sec else 0.0

    key_moments = sentiment.get("key_moments", {})

    most_positive = []
    if key_moments.get("most_positive"):
        mp = key_moments["most_positive"]
        most_positive.append({
            "start": mp.get("start", 0),
            "end": mp.get("end", 0),
            "text": mp.get("text", ""),
            "score": mp.get("score", sentiment.get("overall_score", 0))
        })

    most_negative = []
    if key_moments.get("most_negative"):
        mn = key_moments["most_negative"]
        most_negative.append({
            "start": mn.get("start", 0),
            "end": mn.get("end", 0),
            "text": mn.get("text", ""),
            "score": mn.get("score", sentiment.get("overall_score", 0))
        })

    sentiment_distribution = {
        "positive_pct": safe_pct(sentiment.get("positive_ratio", 0)),
        "neutral_pct": safe_pct(sentiment.get("neutral_ratio", 0)),
        "negative_pct": safe_pct(sentiment.get("negative_ratio", 0)),
    }

    tone_distribution = {}
    for label, value in tone.get("tone_distribution", {}).items():
        tone_distribution[label] = safe_pct(value)

    flagged_instances = []
    for flag in bias.get("bias_flags", []):
        flagged_instances.append({
            "start": flag.get("timestamp_seconds", 0),
            "end": flag.get("timestamp_seconds", 0),
            "sentence": flag.get("text_context", ""),
            "matches": [{
                "keyword": flag.get("keyword", ""),
                "category": flag.get("category", "")
            }],
            "final_score": round(flag.get("confidence", 0.0), 4),
            "level": flag.get("severity", "LOW")
        })

    flagged_moments = []
    for seg in ep.get("flagged_segments", []):
        flagged_moments.append({
            "time_label": seg.get("timestamp", "N/A"),
            "type": seg.get("emotional_state", "Emotional Mismatch"),
            "confidence": f"{seg.get('confidence', 0) * 100:.0f}%",
            "divergence": seg.get("divergence_score", 0)
        })

    payload = {
        "podcast_name": results.get("filename", st.session_state.uploaded_filename or "Unknown"),
        "duration_min": duration_min,
        "transcription": {
            "word_count": word_count,
            "segments": transcript_segments
        },
        "sentiment": {
            "summary": {
                "overall_label": sentiment.get("overall_sentiment", "N/A").upper(),
                "overall_score": sentiment.get("overall_score", "N/A"),
                "confidence": sentiment.get("confidence", "N/A"),
                "sentence_count": sentiment.get("sentence_count", "N/A"),
                "distribution": sentiment_distribution,
                "most_positive": most_positive,
                "most_negative": most_negative
            }
        },
        "tone": {
            "summary": {
                "dominant_tone": tone.get("dominant_tone", "N/A"),
                "tone_score": tone.get("dominant_score", "N/A"),
                "confidence": tone.get("confidence", "N/A"),
                "distribution": tone_distribution
            }
        },
        "bias": {
            "summary": {
                "bias_score": bias.get("overall_bias_score", "N/A"),
                "bias_level": bias.get("bias_level", "N/A"),
                "total_flags": bias.get("bias_flags_count", 0)
            },
            "category_distribution": bias.get("category_distribution", {}),
            "flagged_instances": flagged_instances
        },
        "emotionprint": {
            "summary": {
                "authenticity_pct": f"{ep.get('authenticity_score', 0):.0f}%",
                "mismatch_count": len(ep.get("flagged_segments", [])),
                "sarcasm_count": ep.get("sarcasm_instances", 0),
                "suppression_count": ep.get("suppression_instances", 0),
                "irony_count": ep.get("irony_instances", 0)
            },
            "flagged_moments": flagged_moments
        },
        "transcript_text": transcript_text
    }

    return payload

# ─────────────────────────────────────────────────────────
# Analysis Pipeline
# ─────────────────────────────────────────────────────────
def run_full_analysis(uploaded_file, options: dict) -> dict:
    results = {}
    podcast_id = str(uuid.uuid4())[:8]

    upload_dir = Path(config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    audio_path = upload_dir / f"{podcast_id}_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Audio Preprocessing
    processed_audio_path = convert_to_wav_16k_mono(str(audio_path))
    audio_path = Path(processed_audio_path)

    # Audio Chunking
    chunk_paths = split_audio_into_chunks(
        str(audio_path),
        chunk_length_ms=5 * 60 * 1000
    )

    results["podcast_id"] = podcast_id
    results["audio_path"] = str(audio_path)
    results["filename"] = uploaded_file.name
    results["chunk_paths"] = chunk_paths

    # Database entry
    db = DatabaseManager()
    db.insert_podcast(
        podcast_id=podcast_id,
        filename=audio_path.name,
        original_filename=uploaded_file.name,
        file_size=uploaded_file.size,
        file_path=str(audio_path),
        duration=None
    )

    # Stage 1: Transcription
    with st.status("🎤 Stage 1 of 5 — Transcribing audio...", expanded=True) as status:
        st.write("⚡ Running parallel transcription on chunks...")

        chunk_results_raw = transcribe_chunks_parallel(
            chunk_paths,
            model_size=getattr(config, "WHISPER_MODEL_SIZE", "small"),
            max_workers=2
        )

        # Normalize chunk result structure
        normalized_chunk_results = []
        for idx, item in enumerate(chunk_results_raw):
            if isinstance(item, dict) and "result" in item:
                normalized_chunk_results.append(item)
            else:
                normalized_chunk_results.append({
                    "chunk_path": chunk_paths[idx] if idx < len(chunk_paths) else f"chunk_{idx:03d}.wav",
                    "result": item
                })

        merged_transcript = merge_chunk_transcripts(
            normalized_chunk_results,
            chunk_duration_sec=300
        )

        merged_transcript["word_count"] = len(merged_transcript.get("text", "").split())

        segments = merged_transcript.get("segments", [])
        merged_transcript["duration"] = max([seg.get("end", 0) for seg in segments], default=0)

        transcript_path = Path(config.TRANSCRIPT_DIR) / f"{podcast_id}_transcript.json"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)

        save_merged_transcript(merged_transcript, str(transcript_path))

        transcript = merged_transcript
        results["transcript"] = transcript

        db.update_podcast_status(podcast_id, "transcribed", str(transcript_path))

        status.update(
            label=f"✅ Transcription complete — {transcript.get('word_count', 0)} words",
            state="complete"
        )

    # Stage 2: Sentiment
    with st.status("💬 Stage 2 of 5 — Sentiment analysis...", expanded=False) as status:
        sa = SentimentAnalyzer()
        sentiment = sa.analyze_text(
            transcript["text"],
            transcript.get("segments")
        )
        sa.save_results(
            sentiment,
            str(Path(config.RESULTS_DIR) / f"{podcast_id}_sentiment.json")
        )
        results["sentiment"] = sentiment
        status.update(
            label=f"✅ Sentiment: {sentiment['overall_sentiment'].upper()} "
                  f"(score {sentiment['overall_score']:+.2f})",
            state="complete"
        )

    # Stage 3: Tone
    with st.status("🎭 Stage 3 of 5 — Tone detection...", expanded=False) as status:
        td = ToneDetector()
        tone = td.analyze_text(
            transcript["text"],
            transcript.get("segments")
        )
        td.save_results(
            tone,
            str(Path(config.RESULTS_DIR) / f"{podcast_id}_tone.json")
        )
        results["tone"] = tone
        status.update(
            label=f"✅ Dominant tone: {tone['dominant_tone'].upper()}",
            state="complete"
        )

    # Stage 4: Bias
    with st.status("🔍 Stage 4 of 5 — Bias detection...", expanded=False) as status:
        bd = BiasDetector()
        bias = bd.analyze_text(
            transcript["text"],
            transcript.get("segments"),
            audio_path=str(audio_path) if options.get("extract_audio_context") else None
        )
        bd.save_results(
            bias,
            str(Path(config.RESULTS_DIR) / f"{podcast_id}_bias.json")
        )
        results["bias"] = bias
        status.update(
            label=f"✅ Bias: {bias['bias_level']} "
                  f"({bias['bias_flags_count']} flags detected)",
            state="complete"
        )

    # Stage 5: EmotionPrint
    with st.status("🧠 Stage 5 of 5 — EmotionPrint™ analysis...", expanded=False) as status:
        ep = EmotionPrintAnalyzer()

        segments = transcript.get("segments", [])
        if segments and options.get("run_emotionprint", True):
            ep_results = ep.analyze_full_transcript(
                audio_path=str(audio_path),
                segments=segments,
                sample_every_n=3
            )
        else:
            ep_results = ep._empty_full_result()

        ep.save_results(
            ep_results,
            str(Path(config.RESULTS_DIR) / f"{podcast_id}_emotionprint.json")
        )
        results["emotionprint"] = ep_results

        status.update(
            label=f"✅ EmotionPrint™ — Authenticity: "
                  f"{ep_results['authenticity_score']:.0f}% | "
                  f"Sarcasm: {ep_results['sarcasm_instances']}",
            state="complete"
        )

    db.update_podcast_status(podcast_id, "completed", str(transcript_path))

    return results

# ─────────────────────────────────────────────────────────
# Results Rendering
# ─────────────────────────────────────────────────────────
def render_results(results: dict):
    transcript = results["transcript"]
    sentiment = results["sentiment"]
    tone = results["tone"]
    bias = results["bias"]
    ep = results["emotionprint"]

    st.markdown('<p class="section-header">📊 Analysis Summary</p>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Sentiment",
        sentiment["overall_sentiment"].capitalize(),
        f"{sentiment['overall_score']:+.2f}"
    )
    k2.metric("Dominant Tone", tone["dominant_tone"].capitalize())
    k3.metric("Bias Level", bias["bias_level"], f"{bias['overall_bias_score']:.0f}/100")
    k4.metric("Authenticity", f"{ep['authenticity_score']:.0f}%")
    k5.metric("Duration", f"{transcript.get('duration', 0) / 60:.1f} min")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Sentiment",
        "🎭 Tone",
        "🔍 Bias",
        "🧠 EmotionPrint™",
        "📝 Transcript",
        "📥 Export"
    ])

    # TAB 1: Sentiment
    with tab1:
        st.subheader("Sentiment Analysis")

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score", f"{sentiment['overall_score']:+.3f}")
        c2.metric("Confidence", f"{sentiment['confidence'] * 100:.1f}%")
        c3.metric("Sentences", sentiment["sentence_count"])

        col_a, col_b = st.columns(2)

        with col_a:
            if sentiment.get("timeline"):
                st.plotly_chart(
                    create_sentiment_timeline(sentiment["timeline"]),
                    use_container_width=True
                )

        with col_b:
            st.plotly_chart(
                create_sentiment_distribution_pie(sentiment),
                use_container_width=True
            )

        km = sentiment.get("key_moments", {})
        if km.get("most_positive"):
            st.success(f"🌟 **Most Positive:** _{km['most_positive']['text']}_")
        if km.get("most_negative"):
            st.error(f"⚠️ **Most Negative:** _{km['most_negative']['text']}_")

    # TAB 2: Tone
    with tab2:
        st.subheader("Tone Detection")

        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant Tone", tone["dominant_tone"].capitalize())
        c2.metric("Tone Score", f"{tone['dominant_score']:.2f}")
        c3.metric("Confidence", f"{tone['confidence'] * 100:.1f}%")

        tone_fig = create_tone_distribution_chart(tone)
        if tone_fig is not None:
            st.plotly_chart(tone_fig, use_container_width=True)

        if tone.get("tone_examples"):
            st.write("#### 🎯 Representative Examples per Tone")
            for tone_name, ex in tone["tone_examples"].items():
                with st.expander(f"{tone_name.capitalize()} — score: {ex['score']:.2f}"):
                    st.markdown(f"> _{ex['text']}_")

    # TAB 3: Bias
    with tab3:
        st.subheader("Bias Detection")

        c1, c2, c3 = st.columns(3)
        c1.metric("Bias Score", f"{bias['overall_bias_score']:.1f}/100")
        c2.metric("Bias Level", bias["bias_level"])
        c3.metric("Total Flags", bias["bias_flags_count"])

        col_a, col_b = st.columns(2)

        with col_a:
            if bias.get("timeline"):
                st.plotly_chart(
                    create_bias_timeline(bias["timeline"]),
                    use_container_width=True
                )

        with col_b:
            if bias.get("category_distribution"):
                st.plotly_chart(
                    create_bias_category_chart(bias["category_distribution"]),
                    use_container_width=True
                )

        if bias.get("bias_flags"):
            st.write("#### 🚩 Flagged Instances")

            for i, flag in enumerate(bias["bias_flags"][:20], 1):
                with st.expander(
                    f"#{i} — '{flag['keyword']}' "
                    f"[{flag['category'].replace('_', ' ').upper()}] "
                    f"| {flag.get('timestamp_formatted', 'N/A')}"
                ):
                    st.markdown(f"**Context:**  \n> _{flag['text_context']}_")

                    if flag.get("entities"):
                        ent_text = ", ".join([
                            f"**{e['text']}** ({e['label']})"
                            for e in flag["entities"]
                        ])
                        st.markdown(f"**Entities:** {ent_text}")

                    if flag.get("audio_context_path"):
                        ctx_path = Path(flag["audio_context_path"])
                        if ctx_path.exists():
                            with open(ctx_path, "rb") as audio_f:
                                st.audio(audio_f.read(), format="audio/wav")

    # TAB 4: EmotionPrint
    with tab4:
        st.subheader("EmotionPrint™ — Prosody-Semantic Divergence")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Authenticity", f"{ep['authenticity_score']:.0f}%")
        c2.metric("Sarcasm", ep["sarcasm_instances"])
        c3.metric("Suppression", ep["suppression_instances"])
        c4.metric("Irony", ep["irony_instances"])

        col_a, col_b = st.columns(2)

        with col_a:
            st.plotly_chart(
                create_emotionprint_timeline(ep),
                use_container_width=True
            )

        with col_b:
            st.plotly_chart(
                create_emotionprint_summary_chart(ep),
                use_container_width=True
            )

        if ep.get("flagged_segments"):
            st.write("#### 🎭 Flagged Moments")

            for seg in ep["flagged_segments"][:10]:
                icon = {
                    "Sarcasm": "😏",
                    "Emotional Suppression": "😶",
                    "Irony": "🙃",
                    "Emotional Mismatch": "❓"
                }.get(seg["emotional_state"], "⚠️")

                with st.expander(
                    f"{icon} {seg['emotional_state']} @ {seg['timestamp']} "
                    f"| Confidence: {seg['confidence'] * 100:.0f}% "
                    f"| Divergence: {seg['divergence_score']:.2f}"
                ):
                    st.markdown(f"**Text:** _{seg['text']}_")
                    st.markdown(f"**Explanation:** {seg['explanation']}")

                    pf = seg["prosody_features"]
                    st.markdown(
                        f"**Prosody:** "
                        f"Pitch var: {pf['pitch_variance']:.1f} Hz | "
                        f"Volume: {pf['volume_mean_db']:.1f} dB | "
                        f"Rate: {pf['speech_rate']:.1f} syl/s"
                    )

    # TAB 5: Transcript
    with tab5:
        st.subheader("Color-Coded Transcript")

        st.markdown("""
        **Legend:**  
        🟢 Green = Positive sentiment  
        🔴 Red = Negative sentiment  
        🟠 Orange underline = Bias keyword
        """)

        if sentiment.get("sentences"):
            html = generate_color_coded_transcript(
                sentiment["sentences"],
                bias.get("bias_flags", [])
            )
            st.markdown(html, unsafe_allow_html=True)

    # TAB 6: Export
    with tab6:
        st.subheader("Export Analysis Report")

        ex_col1, ex_col2 = st.columns(2)

        with ex_col1:
            st.write("**📄 JSON Export**")
            st.write("Machine-readable, includes all raw data")

            json_report = {
                "podcast_id": results["podcast_id"],
                "filename": results["filename"],
                "analyzed_at": datetime.now().isoformat(),
                "transcript": results["transcript"],
                "sentiment": results["sentiment"],
                "tone": results["tone"],
                "bias": results["bias"],
                "emotionprint": results["emotionprint"]
            }

            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(json_report, indent=2, ensure_ascii=False),
                file_name=f"vibejudge_{results['podcast_id']}.json",
                mime="application/json",
                key=f"download_json_{results['podcast_id']}"
            )

        with ex_col2:
            st.write("**📑 PDF Report**")
            st.write("Professional report with charts and summary")

            if st.session_state.pdf_bytes is None:
                if st.button("🖨️ Generate PDF Report", key=f"generate_pdf_{results['podcast_id']}"):
                    with st.spinner("Generating PDF..."):
                        try:
                            pdf_payload = build_pdf_payload(results)
                            charts = create_pdf_charts(results)

                            st.session_state.pdf_payload = pdf_payload
                            st.session_state.charts = charts
                            st.session_state.pdf_bytes = generate_analysis_pdf(
                                pdf_payload,
                                charts=charts
                            )

                            st.rerun()
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")
                            logger.exception("PDF generation error")

            else:
                st.success("✅ PDF is ready")
                st.download_button(
                    label="📥 Download PDF",
                    data=st.session_state.pdf_bytes,
                    file_name=f"vibejudge_{results['podcast_id']}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{results['podcast_id']}"
                )

        with st.expander("🔍 Debug PDF Payload"):
            st.json(st.session_state.pdf_payload if st.session_state.pdf_payload else {})

# ─────────────────────────────────────────────────────────
# Page: Analyze
# ─────────────────────────────────────────────────────────
def page_analyze():
    st.markdown('<p class="main-title">🎙️ VibeJudge</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Multimodal Podcast Sentiment, Tone & Bias Analyzer</p>',
        unsafe_allow_html=True
    )
    st.divider()

    st.markdown("### 📁 Upload Podcast")

    uploaded_file = st.file_uploader(
        "Drop your podcast here or click to browse",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        help="Maximum file size: 100 MB | Maximum duration: 60 minutes"
    )

    if uploaded_file:
        is_valid, error_msg = validate_uploaded_file(uploaded_file)

        if not is_valid:
            st.error(f"❌ {error_msg}")
            return

        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.uploaded_file_size = uploaded_file.size

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.success(f"✅ **{uploaded_file.name}**")
        info_col2.info(f"📦 {uploaded_file.size / 1024 / 1024:.1f} MB")
        info_col3.info(f"📄 {Path(uploaded_file.name).suffix.upper()}")

        st.markdown("### ⚙️ Analysis Options")
        opt_col1, opt_col2, opt_col3 = st.columns(3)

        with opt_col1:
            run_emotionprint = st.checkbox(
                "🧠 Run EmotionPrint™",
                value=True,
                help="Detect sarcasm/irony via prosody analysis"
            )
        with opt_col2:
            extract_audio_ctx = st.checkbox(
                "🔊 Extract Audio Context",
                value=False,
                help="Save ±30s audio clips for bias flags (slower)"
            )
        with opt_col3:
            whisper_model = st.selectbox(
                "Whisper Model",
                ["base (faster)", "small (recommended)", "medium (accurate)"],
                index=1
            )

        model_map = {
            "base (faster)": "base",
            "small (recommended)": "small",
            "medium (accurate)": "medium"
        }
        config.WHISPER_MODEL_SIZE = model_map[whisper_model]

        options = {
            "run_emotionprint": run_emotionprint,
            "extract_audio_context": extract_audio_ctx
        }

        st.divider()

        if st.button("🚀 Start Analysis", key="start_analysis_btn"):
            with st.spinner("Running analysis..."):
                try:
                    results = run_full_analysis(uploaded_file, options)

                    st.session_state.analysis_results = results
                    st.session_state.analysis_done = True
                    st.session_state.pdf_bytes = None
                    st.session_state.pdf_payload = None
                    st.session_state.charts = {}

                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    logger.exception("Analysis pipeline error")

    if st.session_state.analysis_done and st.session_state.analysis_results is not None:
        render_results(st.session_state.analysis_results)

# ─────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────
def page_dashboard():
    st.title("📊 Analysis Dashboard")

    db = DatabaseManager()
    recent = db.get_recent_podcasts(limit=10)

    if not recent:
        st.info("No podcasts analyzed yet. Go to **Analyze** to get started!")
        return

    stats = db.get_statistics()

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Analyzed", stats.get("total_podcasts", 0))
    s2.metric("Completed", stats.get("completed", 0))
    s3.metric("This Week", stats.get("this_week", 0))

    st.divider()
    st.write("### Recent Analyses")

    for pod in recent:
        with st.expander(
            f"🎙️ {pod['original_filename']} | "
            f"{pod['status'].upper()} | "
            f"{pod['upload_date'][:10]}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**ID:** {pod.get('podcast_id', 'N/A')}")
            c2.write(
                f"**Duration:** {pod['duration'] / 60:.1f} min"
                if pod.get("duration") else "N/A"
            )
            c3.write(f"**Size:** {pod.get('file_size', 0) / 1024 / 1024:.1f} MB")

            pod_id = pod.get("podcast_id") or pod.get("id") or "unknown"
            result_path = Path(config.RESULTS_DIR) / f"{pod_id}_sentiment.json"
            if result_path.exists():
                with open(result_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                st.write(
                    f"**Sentiment:** "
                    f"{cached.get('overall_sentiment', 'N/A').upper()} "
                    f"({cached.get('overall_score', 0):+.2f})"
                )

# ─────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────
def page_about():
    st.title("ℹ️ About VibeJudge")

    st.markdown("""
    ## What is VibeJudge?

    **VibeJudge** is a multimodal batch-mode podcast analysis platform
    combining speech recognition, NLP, and acoustic prosody analysis
    to reveal the sentiment, tone, and bias hidden in spoken content.

    ## Core Innovations

    | Feature | Description |
    |---------|-------------|
    | **TCBD** | Temporal Context-Aware Bias Detection with ±30s audio clips |
    | **EmotionPrint™** | Prosody-semantic divergence for sarcasm detection |
    | **Adaptive Chunking** | Sentence-boundary-preserving audio segmentation |
    | **Explainable AI** | User-extensible dictionaries with crowdsourced validation |

    ## Technology Stack

    - **ASR:** OpenAI Whisper / Faster-Whisper
    - **Sentiment:** Cardiff NLP RoBERTa
    - **Prosody:** librosa acoustic analysis
    - **NER:** spaCy en_core_web_sm
    - **Visualization:** Plotly
    - **UI:** Streamlit

    ## Team
    - Puneeth Sai Goutam (23071A6635)
    - Hasini Reddy (23071A6641)
    - Pankaj (23071A6643)
    - Srinivas (23071A6646)
    - Akanksha Bhosle (24075A6601)

    **Guide:** [Your Guide Name]  
    **Version:** 1.3
    """)

# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────
def main():
    page = render_sidebar()

    if page == "🏠 Analyze":
        page_analyze()
    elif page == "📊 Dashboard":
        page_dashboard()
    else:
        page_about()

if __name__ == "__main__":
    main()
