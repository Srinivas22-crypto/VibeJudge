import os
import streamlit as st
import pandas as pd
from models.transcriber import Transcriber
from models.sentiment_analyzer import SentimentAnalyzer
from models.bias_detector import BiasDetector
from database.db_manager import DatabaseManager
from config.settings import Config
import uuid, json
from datetime import datetime

config = Config()
db = DatabaseManager()

def render_batch_processor():
    """Streamlit component: upload and process multiple podcasts at once."""
    st.header("🎙️ Batch Processor")
    st.markdown("Upload **multiple podcast files** for simultaneous analysis.")

    uploaded_files = st.file_uploader(
        "Select multiple audio files",
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        accept_multiple_files=True,
        help="Max 100 MB per file, up to 10 files"
    )

    if not uploaded_files:
        st.info("👆 Upload one or more audio files to begin batch processing.")
        return

    st.write(f"**{len(uploaded_files)} files selected:**")
    file_info = []
    for uf in uploaded_files:
        size_mb = len(uf.getvalue()) / 1e6
        valid = size_mb <= 100
        file_info.append({
            "Filename": uf.name,
            "Size (MB)": round(size_mb, 2),
            "Valid": "✅" if valid else "❌ Too large"
        })
    st.dataframe(pd.DataFrame(file_info))

    if st.button("🚀 Start Batch Analysis", type="primary"):
        valid_files = [uf for uf in uploaded_files if len(uf.getvalue()) / 1e6 <= 100]
        if not valid_files:
            st.error("No valid files to process.")
            return

        results = []
        progress = st.progress(0, text="Initializing...")
        transcriber = Transcriber()
        analyzer = SentimentAnalyzer()
        detector = BiasDetector()

        for i, uf in enumerate(valid_files):
            progress.progress((i) / len(valid_files),
                               text=f"Processing {uf.name} ({i+1}/{len(valid_files)})...")
            podcast_id = str(uuid.uuid4())
            save_path = os.path.join(config.UPLOAD_DIR, f"{podcast_id}.{uf.name.rsplit('.',1)[-1]}")
            os.makedirs(config.UPLOAD_DIR, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uf.getvalue())

            try:
                trans = transcriber.transcribe(save_path)
                text = trans["text"]
                sentiment = analyzer.analyze_text(text)
                bias = detector.analyze_text(text)

                results.append({
                    "podcast_id": podcast_id,
                    "filename": uf.name,
                    "duration": round(trans.get("duration", 0), 1),
                    "overall_sentiment": sentiment.get("overall_sentiment", "N/A"),
                    "bias_score": bias.get("overall_bias_score", 0),
                    "bias_level": bias.get("bias_level", "N/A"),
                    "word_count": len(text.split()),
                    "status": "✅ Done"
                })
            except Exception as e:
                results.append({
                    "podcast_id": podcast_id,
                    "filename": uf.name,
                    "status": f"❌ Error: {str(e)[:50]}"
                })

        progress.progress(1.0, text="✅ Batch complete!")
        st.success(f"Processed {len(results)} files!")

        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Export results as CSV
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Batch Report (CSV)",
            data=csv,
            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
