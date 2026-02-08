import streamlit as st
import time
import os
import json
from pathlib import Path


# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import components and models
from config.settings import APP_TITLE, APP_ICON, PAGE_LAYOUT, UPLOAD_DIR
from app.components.uploader import render_upload_sectionstreamlit run app/main.py 
from models.transcriber import get_transcriber
from models.analyzer import get_analyzer
from database.db_manager import get_db

# Page Config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .bias-high { border-left: 5px solid #ff4b4b; }
    .bias-moderate { border-left: 5px solid #ffa500; }
    .bias-low { border-left: 5px solid #00c853; }
</style>
""", unsafe_allow_html=True)

def main():
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.markdown("Upload a podcast episode to detect bias, sentiment, and tone.")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **VibeJudge** analyzes audio content using AI to provide insights on:
        - 🎭 **Sentiment**: Positive/Negative balance
        - ⚖️ **Bias**: Loaded language and political leaning
        - 🗣️ **Tone**: Aggression, excitement, or calm
        """)
        
        st.divider()
        db = get_db()
        stats = db.get_statistics()
        st.subheader("Global Stats")
        st.metric("Podcasts Analyzed", stats['total_podcasts'])
        st.metric("Avg Bias Score", stats['avg_bias_score'])

    # --- Main Content ---
    
    # 1. Upload Section
    if 'current_podcast' not in st.session_state:
        upload_result = render_upload_section()
        if upload_result:
            uploaded_file, podcast_id, file_path, duration = upload_result
            
            # Store in session
            st.session_state['current_podcast'] = {
                'id': podcast_id,
                'path': file_path,
                'filename': uploaded_file.name,
                'duration': duration
            }
            
            # Save to DB
            db.insert_podcast(
                podcast_id, uploaded_file.name, uploaded_file.name, 
                uploaded_file.size, file_path, duration
            )
            st.rerun()

    # 2. Analysis Section
    else:
        podcast = st.session_state['current_podcast']
        st.info(f"Analyzing: **{podcast['filename']}**")
        
        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Transcription
            status_text.text("🎙️ Transcribing audio (this may take a while)...")
            progress_bar.progress(10)
            
            transcriber = get_transcriber()
            
            # Preprocess
            processed_path = transcriber.preprocess_audio(podcast['path'])
            
            # Transcribe
            transcript_result = transcriber.transcribe(processed_path)
            progress_bar.progress(50)
            
            # Save Transcript
            transcript_path = Path(podcast['path']).with_suffix('.json')
            transcriber.save_transcript(transcript_result, str(transcript_path))
            
            full_text = transcript_result['text']
            
            # Step 2: Analysis
            status_text.text("🧠 Analyzing content for sentiment and bias...")
            progress_bar.progress(60)
            
            analyzer = get_analyzer()
            
            # Run Analyses
            sentiment = analyzer.analyze_sentiment(full_text)
            bias = analyzer.analyze_bias(full_text)
            tone = analyzer.analyze_tone(full_text)
            
            progress_bar.progress(90)
            
            # Step 3: Save Results
            status_text.text("💾 Saving results...")
            
            # Processing time (mock)
            processing_time = 0 
            
            # Save to DB
            analysis_id = db.insert_analysis(
                podcast['id'], sentiment, tone, bias, 
                processing_time=120.0,  # Placeholder
                result_json_path=str(transcript_path)
            )
            
            # Save Flags
            if bias['flags']:
                db.insert_bias_flags(analysis_id, bias['flags'])
            
            db.update_podcast_status(podcast['id'], 'completed')
            
            progress_bar.progress(100)
            status_text.success("Analysis Complete!")
            time.sleep(1)
            
            # --- Results Dashboard ---
            st.divider()
            
            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Overall Sentiment", 
                         f"{sentiment['overall_score']}", 
                         delta=f"{sentiment['positive_pct']}% Positive")
            with c2:
                level_color = "🟢" if bias['level'] == "Low" else "🟠" if bias['level'] == "Moderate" else "🔴"
                st.metric("Bias Level", f"{level_color} {bias['level']}", f"{bias['score']} Score")
            with c3:
                st.metric("Dominant Tone", tone['dominant_tone'])
            
            # Tabs for details
            tab1, tab2, tab3 = st.tabs(["📝 Transcript", "🚩 Bias Flags", "📊 Data"])
            
            with tab1:
                st.text_area("Full Transcript", full_text, height=300)
            
            with tab2:
                if bias['flags']:
                    st.warning(f"Found {len(bias['flags'])} potential bias indicators:")
                    for flag in bias['flags']:
                        st.markdown(f"- **{flag['phrase']}** ({flag['category']})")
                else:
                    st.success("No significant bias flags detected.")
                    
            with tab3:
                st.json(sentiment)
                st.json(tone)
            
            # Button to reset
            if st.button("Analyze Another File"):
                del st.session_state['current_podcast']
                st.rerun()

        except Exception as e:
            st.error(f"Analysis Failed: {e}")
            db.update_podcast_status(podcast['id'], 'failed', str(e))
            if st.button("Try Again"):
                del st.session_state['current_podcast']
                st.rerun()

if __name__ == "__main__":
    main()
