import streamlit as st
import time
import json
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Imports
from config.settings import APP_TITLE, APP_ICON, PAGE_LAYOUT, RESULTS_DIR
from app.components.uploader import render_upload_section
from models.transcriber import get_transcriber
from models.analyzer import get_analyzer
from models.sentiment_analyzer import SentimentAnalyzer
from models.tone_detector import ToneDetector
from database.db_manager import get_db
from utils.visualizations import (
    create_sentiment_timeline,
    create_sentiment_distribution_pie,
    create_tone_heatmap,
    create_combined_dashboard
)
from utils.pdf_generator import generate_pdf_report


# Page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT
)


def main():
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("Upload a podcast episode to detect bias, sentiment, and tone.")

    # ================= SIDEBAR =================
    with st.sidebar:
        st.header("About")
        st.markdown("""
**VibeJudge analyzes audio using AI**
- 🎭 Sentiment
- ⚖️ Bias
- 🗣️ Emotional Tone
""")

        db = get_db()
        stats = db.get_statistics()
        st.metric("Podcasts Analyzed", stats['total_podcasts'])
        st.metric("Avg Bias Score", stats['avg_bias_score'])

    # ================= UPLOAD =================
    if 'current_podcast' not in st.session_state:
        upload_result = render_upload_section()
        if upload_result:
            uploaded_file, podcast_id, file_path, duration = upload_result

            st.session_state['current_podcast'] = {
                'id': podcast_id,
                'path': file_path,
                'filename': uploaded_file.name,
                'duration': duration
            }

            db.insert_podcast(
                podcast_id,
                uploaded_file.name,
                uploaded_file.name,
                uploaded_file.size,
                file_path,
                duration
            )
            st.rerun()

    # ================= ANALYSIS =================
    else:
        podcast = st.session_state['current_podcast']
        st.info(f"Analyzing: **{podcast['filename']}**")

        progress_bar = st.progress(0)
        status = st.empty()

        try:
            # -----------------------------
            # STEP 1 — TRANSCRIBE
            # -----------------------------
            status.text("🎙️ Transcribing audio...")
            progress_bar.progress(10)

            transcriber = get_transcriber()
            processed = transcriber.preprocess_audio(podcast['path'])
            transcript_result = transcriber.transcribe(processed)

            transcript_path = Path(podcast['path']).with_suffix(".json")
            transcriber.save_transcript(transcript_result, str(transcript_path))
            full_text = transcript_result["text"]

            progress_bar.progress(40)

            # -----------------------------
            # STEP 2 — SENTIMENT
            # -----------------------------
            sentiment_model = SentimentAnalyzer()
            sentiment_results = sentiment_model.analyze_text(
                full_text,
                transcript_result.get("segments")
            )

            sentiment_path = Path(RESULTS_DIR) / f"{podcast['id']}_sentiment.json"
            sentiment_model.save_results(sentiment_results, str(sentiment_path))
            progress_bar.progress(60)

            # -----------------------------
            # STEP 3 — TONE
            # -----------------------------
            tone_detector = ToneDetector()
            tone_results = tone_detector.analyze_text(
                full_text,
                transcript_result.get("segments")
            )

            tone_path = Path(RESULTS_DIR) / f"{podcast['id']}_tone.json"
            tone_detector.save_results(tone_results, str(tone_path))
            progress_bar.progress(75)

            # -----------------------------
            # STEP 4 — BIAS
            # -----------------------------
            analyzer = get_analyzer()
            bias = analyzer.analyze_bias(full_text)
            progress_bar.progress(90)

            # -----------------------------
            # STEP 5 — SAVE DB
            # -----------------------------
            analysis_id = db.insert_analysis(
                podcast['id'],
                sentiment_results,
                tone_results,
                bias,
                processing_time=120.0,
                result_json_path=str(transcript_path)
            )

            if bias['flags']:
                db.insert_bias_flags(analysis_id, bias['flags'])

            progress_bar.progress(100)
            status.success("Analysis Complete!")

            # ==================================================
            # STEP 6 — VISUAL DASHBOARD
            # ==================================================
            st.write("---")
            st.write("## 📊 Analysis Dashboard")

            try:
                dashboard_fig = create_combined_dashboard(
                    sentiment_results,
                    tone_results
                )
                st.plotly_chart(dashboard_fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Combined dashboard failed: {e}")

                col1, col2 = st.columns(2)

                with col1:
                    if sentiment_results.get("timeline"):
                        st.plotly_chart(
                            create_sentiment_timeline(sentiment_results["timeline"]),
                            use_container_width=True
                        )

                with col2:
                    st.plotly_chart(
                        create_sentiment_distribution_pie(sentiment_results),
                        use_container_width=True
                    )

                if tone_results.get("timeline"):
                    st.plotly_chart(
                        create_tone_heatmap(tone_results["timeline"]),
                        use_container_width=True
                    )

            # ==================================================
            # TRANSCRIPT HIGHLIGHTS
            # ==================================================
            st.write("---")
            st.write("## 📝 Full Transcript with Sentiment Highlights")

            if sentiment_results.get("sentences"):
                html = "<div style='line-height:2;'>"
                for s in sentiment_results["sentences"][:50]:
                    score = abs(s["score"])
                    if s["label"] == "positive":
                        color = f"rgba(46,204,113,{score*0.3})"
                    elif s["label"] == "negative":
                        color = f"rgba(231,76,60,{score*0.3})"
                    else:
                        color = "transparent"

                    html += f"<span style='background:{color};padding:2px 4px;border-radius:3px;'>{s['text']}</span> "

                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)

            # ==================================================
            # STEP 7 — EXPORT REPORT
            # ==================================================
            st.write("---")
            st.write("## 📥 Export Report")

            export_col1, export_col2 = st.columns(2)

            # JSON EXPORT
            with export_col1:
                json_report = {
                    "podcast_id": podcast['id'],
                    "filename": podcast['filename'],
                    "transcript": transcript_result,
                    "sentiment": sentiment_results,
                    "tone": tone_results,
                    "analysis_date": datetime.now().isoformat()
                }

                st.download_button(
                    "📄 Download JSON Report",
                    json.dumps(json_report, indent=2),
                    file_name=f"{podcast['id']}_report.json",
                    mime="application/json"
                )

            # PDF EXPORT
            with export_col2:
                if st.button("📑 Generate PDF Report"):
                    with st.spinner("Generating PDF..."):
                        try:
                            pdf_path = Path(RESULTS_DIR) / f"{podcast['id']}_report.pdf"

                            generate_pdf_report(
                                podcast_id=podcast['id'],
                                filename=podcast['filename'],
                                transcript_data=transcript_result,
                                sentiment_results=sentiment_results,
                                tone_results=tone_results,
                                output_path=str(pdf_path)
                            )

                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "📥 Download PDF",
                                    f,
                                    file_name=f"{podcast['id']}_report.pdf",
                                    mime="application/pdf"
                                )

                            st.success("✓ PDF generated!")

                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")

            # Final DB update
            db.update_podcast_status(
                podcast_id=podcast['id'],
                status="completed",
                transcript_path=str(transcript_path)
            )

            st.success("🎉 Analysis Complete!")
            st.balloons()

            if st.button("Analyze Another File"):
                del st.session_state['current_podcast']
                st.rerun()

        except Exception as e:
            st.error(f"Analysis Failed: {e}")
            db.update_podcast_status(podcast['id'], 'failed', str(e))


if __name__ == "__main__":
    main()