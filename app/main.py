# File: app/main.py  (Complete Week 3 Version)

"""
VibeJudge - Main Streamlit Application
Week 3: Full Pipeline Integration
"""

import json
import uuid
import logging
import sys
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────
# Path Resolution Fix (Streamlit)
# ─────────────────────────────────────────────────────────
# Add project root to sys.path to allow imports from sibling packages
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import streamlit as st
import plotly.graph_objects as go

from models.transcriber import Transcriber
from models.sentiment_analyzer import SentimentAnalyzer
from models.tone_detector import ToneDetector
from models.bias_detector import BiasDetector
from models.emotion_print import EmotionPrintAnalyzer
from database.db_manager import DatabaseManager
from utils.visualizations import (
    create_sentiment_timeline,
    create_sentiment_distribution_pie,
    create_tone_heatmap,
    create_combined_dashboard,
    create_bias_timeline,
    create_bias_category_chart,
    create_emotionprint_timeline,
    create_emotionprint_summary_chart,
    generate_color_coded_transcript
)
from utils.pdf_generator import generate_pdf_report
from config import settings as config

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

# Custom CSS
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
    """Render sidebar and return selected page"""
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
    """
    Validate uploaded file.

    Returns:
        (is_valid: bool, error_message: str)
    """
    # Format check
    allowed = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in allowed:
        return False, f"Unsupported format '{ext}'. Use: {', '.join(allowed)}"

    # Size check (100 MB)
    if uploaded_file.size > 100 * 1024 * 1024:
        size_mb = uploaded_file.size / 1024 / 1024
        return False, f"File too large ({size_mb:.1f} MB). Maximum is 100 MB."

    return True, ""


# ─────────────────────────────────────────────────────────
# Analysis Pipeline
# ─────────────────────────────────────────────────────────
def run_full_analysis(uploaded_file, options: dict) -> dict:
    """
    Run complete analysis pipeline.

    Args:
        uploaded_file : Streamlit UploadedFile object
        options       : Dict of user-selected analysis options

    Returns:
        Dict containing all results
    """
    results = {}
    podcast_id = str(uuid.uuid4())[:8]

    # ── Save uploaded file ──────────────────────────────
    upload_dir = Path(config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    audio_path = upload_dir / f"{podcast_id}_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    results["podcast_id"] = podcast_id
    results["audio_path"] = str(audio_path)
    results["filename"]   = uploaded_file.name

    # ── Database entry ──────────────────────────────────
    db = DatabaseManager()
    db.insert_podcast(
        podcast_id=podcast_id,
        filename=audio_path.name,
        original_filename=uploaded_file.name,
        file_size=uploaded_file.size,
        duration=None
    )

    # ── Stage 1: Transcription ──────────────────────────
    with st.status("🎤 Stage 1 of 5 — Transcribing audio...", expanded=True) as status:
        st.write("Loading Whisper model...")
        transcriber = Transcriber(model_size=config.WHISPER_MODEL_SIZE)

        st.write("Transcribing audio (this may take a moment)...")
        transcript = transcriber.transcribe(str(audio_path), word_timestamps=True)

        transcript_path = (
            Path(config.TRANSCRIPTS_DIR) / f"{podcast_id}_transcript.json"
        )
        transcriber.save_transcript(transcript, str(transcript_path))
        results["transcript"] = transcript

        db.update_podcast_status(
            podcast_id, "transcribed", str(transcript_path)
        )

        status.update(
            label=f"✅ Transcription complete — {transcript['word_count']} words",
            state="complete"
        )

    # ── Stage 2: Sentiment Analysis ─────────────────────
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

    # ── Stage 3: Tone Detection ──────────────────────────
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

    # ── Stage 4: Bias Detection ──────────────────────────
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

    # ── Stage 5: EmotionPrint™ ───────────────────────────
    with st.status("🧠 Stage 5 of 5 — EmotionPrint™ analysis...", expanded=False) as status:
        ep = EmotionPrintAnalyzer()

        segments = transcript.get("segments", [])
        if segments and options.get("run_emotionprint", True):
            ep_results = ep.analyze_full_transcript(
                audio_path=str(audio_path),
                segments=segments,
                sample_every_n=3  # Balance speed vs coverage
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

    # ── Update final DB status ───────────────────────────
    db.update_podcast_status(podcast_id, "completed", str(transcript_path))

    return results


# ─────────────────────────────────────────────────────────
# Results Rendering
# ─────────────────────────────────────────────────────────
def render_results(results: dict):
    """Render all analysis results in tabs"""

    transcript = results["transcript"]
    sentiment  = results["sentiment"]
    tone       = results["tone"]
    bias       = results["bias"]
    ep         = results["emotionprint"]

    # ── Top-level KPIs ───────────────────────────────────
    st.markdown('<p class="section-header">📊 Analysis Summary</p>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Sentiment",
        sentiment["overall_sentiment"].capitalize(),
        f"{sentiment['overall_score']:+.2f}"
    )
    k2.metric(
        "Dominant Tone",
        tone["dominant_tone"].capitalize()
    )
    k3.metric(
        "Bias Level",
        bias["bias_level"],
        f"{bias['overall_bias_score']:.0f}/100"
    )
    k4.metric(
        "Authenticity",
        f"{ep['authenticity_score']:.0f}%"
    )
    k5.metric(
        "Duration",
        f"{transcript.get('duration', 0)/60:.1f} min"
    )

    st.divider()

    # ── Tabs ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Sentiment",
        "🎭 Tone",
        "🔍 Bias",
        "🧠 EmotionPrint™",
        "📝 Transcript",
        "📥 Export"
    ])

    # ────────── TAB 1: SENTIMENT ─────────────────────────
    with tab1:
        st.subheader("Sentiment Analysis")

        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score",    f"{sentiment['overall_score']:+.3f}")
        c2.metric("Confidence",       f"{sentiment['confidence']*100:.1f}%")
        c3.metric("Sentences",        sentiment["sentence_count"])

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

        # Key moments
        km = sentiment.get("key_moments", {})
        if km.get("most_positive"):
            st.success(f"🌟 **Most Positive:** _{km['most_positive']['text']}_")
        if km.get("most_negative"):
            st.error(f"⚠️ **Most Negative:** _{km['most_negative']['text']}_")

    # ────────── TAB 2: TONE ──────────────────────────────
    with tab2:
        st.subheader("Tone Detection")

        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant Tone",  tone["dominant_tone"].capitalize())
        c2.metric("Tone Score",     f"{tone['dominant_score']:.2f}")
        c3.metric("Confidence",     f"{tone['confidence']*100:.1f}%")

        if tone.get("tone_distribution"):
            labels = list(tone["tone_distribution"].keys())
            values = [tone["tone_distribution"][l] * 100 for l in labels]

            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=values,
                marker_color=[
                    "#3498db","#e74c3c","#f39c12",
                    "#9b59b6","#2ecc71","#1abc9c"
                ][:len(labels)]
            )])
            fig.update_layout(
                title="Tone Distribution (%)",
                xaxis_title="Tone",
                yaxis_title="Percentage",
                template="plotly_white",
                height=380
            )
            st.plotly_chart(fig, use_container_width=True)

        if tone.get("tone_examples"):
            st.write("#### 🎯 Representative Examples per Tone")
            for tone_name, ex in tone["tone_examples"].items():
                with st.expander(
                    f"{tone_name.capitalize()} — score: {ex['score']:.2f}"
                ):
                    st.markdown(f"> _{ex['text']}_")

    # ────────── TAB 3: BIAS ──────────────────────────────
    with tab3:
        st.subheader("Bias Detection")

        c1, c2, c3 = st.columns(3)
        c1.metric("Bias Score",  f"{bias['overall_bias_score']:.1f}/100")
        c2.metric("Bias Level",  bias["bias_level"])
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

        # Bias flags table
        if bias.get("bias_flags"):
            st.write("#### 🚩 Flagged Instances")

            for i, flag in enumerate(bias["bias_flags"][:20], 1):
                sev_color = {
                    "HIGH":   "badge-high",
                    "MEDIUM": "badge-medium",
                    "LOW":    "badge-low"
                }.get(flag["severity"], "")

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

            if bias["bias_flags_count"] > 20:
                st.info(
                    f"Showing 20 of {bias['bias_flags_count']} flags. "
                    "Download JSON for complete list."
                )

    # ────────── TAB 4: EMOTIONPRINT™ ─────────────────────
    with tab4:
        st.subheader("EmotionPrint™ — Prosody-Semantic Divergence")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Authenticity",  f"{ep['authenticity_score']:.0f}%")
        c2.metric("Sarcasm",       ep["sarcasm_instances"])
        c3.metric("Suppression",   ep["suppression_instances"])
        c4.metric("Irony",         ep["irony_instances"])

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

        # Flagged segments
        if ep.get("flagged_segments"):
            st.write("#### 🎭 Flagged Moments")

            for seg in ep["flagged_segments"][:10]:
                icon = {
                    "Sarcasm":              "😏",
                    "Emotional Suppression":"😶",
                    "Irony":                "🙃",
                    "Emotional Mismatch":   "❓"
                }.get(seg["emotional_state"], "⚠️")

                with st.expander(
                    f"{icon} {seg['emotional_state']} @ {seg['timestamp']} "
                    f"| Confidence: {seg['confidence']*100:.0f}% "
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

        # Key moments
        km = ep.get("key_moments", {})
        if km.get("highest_sarcasm"):
            hs = km["highest_sarcasm"]
            st.info(
                f"🏆 **Highest-Confidence Sarcasm** @ {hs['timestamp']}: "
                f"_{hs['text'][:100]}_  (confidence: {hs['confidence']*100:.0f}%)"
            )

    # ────────── TAB 5: TRANSCRIPT ────────────────────────
    with tab5:
        st.subheader("Color-Coded Transcript")

        st.markdown("""
        **Legend:**
        🟢 Green = Positive sentiment &nbsp;&nbsp;
        🔴 Red = Negative sentiment &nbsp;&nbsp;
        🟠 Orange underline = Bias keyword
        """)

        if sentiment.get("sentences"):
            html = generate_color_coded_transcript(
                sentiment["sentences"],
                bias.get("bias_flags", [])
            )
            st.markdown(html, unsafe_allow_html=True)

            if len(sentiment["sentences"]) > 60:
                st.info(
                    f"Showing all {len(sentiment['sentences'])} sentences. "
                    "Scroll to view full transcript."
                )

    # ────────── TAB 6: EXPORT ────────────────────────────
    with tab6:
        st.subheader("Export Analysis Report")

        ex_col1, ex_col2 = st.columns(2)

        with ex_col1:
            st.write("**📄 JSON Export**")
            st.write("Machine-readable, includes all raw data")

            json_report = {
                "podcast_id":  results["podcast_id"],
                "filename":    results["filename"],
                "analyzed_at": datetime.now().isoformat(),
                "transcript":  results["transcript"],
                "sentiment":   results["sentiment"],
                "tone":        results["tone"],
                "bias":        results["bias"],
                "emotionprint":results["emotionprint"]
            }

            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(json_report, indent=2),
                file_name=f"vibejudge_{results['podcast_id']}.json",
                mime="application/json"
            )

        with ex_col2:
            st.write("**📑 PDF Report**")
            st.write("Professional report with charts and summary")

            if st.button("🖨️ Generate PDF Report"):
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_path = (
                            Path(config.RESULTS_DIR) /
                            f"{results['podcast_id']}_report.pdf"
                        )
                        generate_pdf_report(
                            podcast_id=results["podcast_id"],
                            filename=results["filename"],
                            transcript_data=results["transcript"],
                            sentiment_results=results["sentiment"],
                            tone_results=results["tone"],
                            output_path=str(pdf_path)
                        )

                        with open(pdf_path, "rb") as pf:
                            st.download_button(
                                label="📥 Download PDF",
                                data=pf.read(),
                                file_name=f"vibejudge_{results['podcast_id']}.pdf",
                                mime="application/pdf"
                            )

                        st.success("✅ PDF report ready!")

                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")


# ─────────────────────────────────────────────────────────
# Page: Analyze
# ─────────────────────────────────────────────────────────
def page_analyze():
    """Main analysis page"""
    st.markdown('<p class="main-title">🎙️ VibeJudge</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Multimodal Podcast Sentiment, Tone & Bias Analyzer</p>',
        unsafe_allow_html=True
    )
    st.divider()

    # Upload section
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

        # File info
        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.success(f"✅ **{uploaded_file.name}**")
        info_col2.info(f"📦 {uploaded_file.size/1024/1024:.1f} MB")
        info_col3.info(f"📄 {Path(uploaded_file.name).suffix.upper()}")

        # Analysis options
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

        # Map model selection
        model_map = {
            "base (faster)":         "base",
            "small (recommended)":   "small",
            "medium (accurate)":     "medium"
        }
        config.WHISPER_MODEL_SIZE = model_map[whisper_model]

        options = {
            "run_emotionprint":      run_emotionprint,
            "extract_audio_context": extract_audio_ctx
        }

        st.divider()

        # Analyze button
        if st.button(
            "🚀 Start Analysis",
            type="primary",
            use_container_width=True
        ):
            try:
                results = run_full_analysis(uploaded_file, options)
                st.success("🎉 **Analysis Complete!** Explore results in the tabs below.")
                st.balloons()
                render_results(results)

            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                logger.error(f"Analysis error: {e}", exc_info=True)


# ─────────────────────────────────────────────────────────
# Page: Dashboard
# ─────────────────────────────────────────────────────────
def page_dashboard():
    """Dashboard page showing recent analyses"""
    st.title("📊 Analysis Dashboard")

    db = DatabaseManager()
    recent = db.get_recent_podcasts(limit=10)

    if not recent:
        st.info("No podcasts analyzed yet. Go to **Analyze** to get started!")
        return

    stats = db.get_statistics()

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Analyzed", stats.get("total_podcasts", 0))
    s2.metric("Completed",      stats.get("completed",     0))
    s3.metric("This Week",      stats.get("this_week",     0))

    st.divider()
    st.write("### Recent Analyses")

    for pod in recent:
        with st.expander(
            f"🎙️ {pod['original_filename']} | "
            f"{pod['status'].upper()} | "
            f"{pod['upload_date'][:10]}"
        ):
            c1, c2, c3 = st.columns(3)
            c1.write(f"**ID:** {pod['podcast_id']}")
            c2.write(
                f"**Duration:** "
                f"{pod['duration']/60:.1f} min" if pod["duration"] else "N/A"
            )
            c3.write(
                f"**Size:** "
                f"{pod['file_size']/1024/1024:.1f} MB"
            )

            result_path = (
                Path(config.RESULTS_DIR) /
                f"{pod['podcast_id']}_sentiment.json"
            )
            if result_path.exists():
                with open(result_path) as f:
                    cached = json.load(f)
                st.write(
                    f"**Sentiment:** "
                    f"{cached['overall_sentiment'].upper()} "
                    f"({cached['overall_score']:+.2f})"
                )


# ─────────────────────────────────────────────────────────
# Page: About
# ─────────────────────────────────────────────────────────
def page_about():
    """About page"""
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

    - **ASR:** OpenAI Whisper
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
    **Version:** 1.3 — Week 3 Build
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
