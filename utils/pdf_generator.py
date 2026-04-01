# utils/pdf_generator.py

import os
import tempfile
from datetime import datetime
from fpdf import FPDF


class AnalysisPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "VibeJudge Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, str(text))
        self.ln(1)

    def kv_row(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.cell(55, 7, f"{key}:", border=0)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")

    def add_plotly_figure(self, fig, title="Chart"):
        self.section_title(title)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp_path = tmp.name
        fig.write_image(tmp_path, format="png", scale=2)

        self.image(tmp_path, w=180)
        self.ln(4)

        try:
            os.remove(tmp_path)
        except Exception:
            pass


def generate_analysis_pdf(analysis_data, charts=None):
    """
    analysis_data structure example:
    {
      "podcast_name": "sample.mp3",
      "duration_min": 3.7,
      "transcription": {...},
      "sentiment": {...},
      "tone": {...},
      "bias": {...},
      "emotionprint": {...},
      "transcript_text": "...."
    }

    charts = {
      "sentiment_timeline": fig1,
      "sentiment_distribution": fig2,
      "tone_distribution": fig3,
      "bias_timeline": fig4,
      "bias_distribution": fig5,
      "emotion_timeline": fig6,
      "emotion_summary": fig7
    }
    """
    pdf = AnalysisPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ---------------- COVER PAGE ----------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "VibeJudge Podcast Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    pdf.section_title("Podcast Metadata")
    pdf.kv_row("Podcast Name", analysis_data.get("podcast_name", "Unknown"))
    pdf.kv_row("Duration", f"{analysis_data.get('duration_min', 0)} min")
    pdf.kv_row("Word Count", analysis_data.get("transcription", {}).get("word_count", "N/A"))
    pdf.kv_row("Sentence Count", analysis_data.get("sentiment", {}).get("summary", {}).get("sentence_count", "N/A"))

    # ---------------- EXECUTIVE SUMMARY ----------------
    pdf.ln(4)
    pdf.section_title("Executive Summary")

    sentiment_summary = analysis_data.get("sentiment", {}).get("summary", {})
    tone_summary = analysis_data.get("tone", {}).get("summary", {})
    bias_summary = analysis_data.get("bias", {}).get("summary", {})
    emotion_summary = analysis_data.get("emotionprint", {}).get("summary", {})

    pdf.kv_row("Overall Sentiment", sentiment_summary.get("overall_label", "N/A"))
    pdf.kv_row("Sentiment Score", sentiment_summary.get("overall_score", "N/A"))
    pdf.kv_row("Dominant Tone", tone_summary.get("dominant_tone", "N/A"))
    pdf.kv_row("Tone Confidence", tone_summary.get("confidence", "N/A"))
    pdf.kv_row("Bias Score", bias_summary.get("bias_score", "N/A"))
    pdf.kv_row("Bias Level", bias_summary.get("bias_level", "N/A"))
    pdf.kv_row("Bias Flags", bias_summary.get("total_flags", "N/A"))
    pdf.kv_row("Authenticity", emotion_summary.get("authenticity_pct", "N/A"))
    pdf.kv_row("Mismatch Count", emotion_summary.get("mismatch_count", "N/A"))

    # ---------------- SENTIMENT SECTION ----------------
    pdf.add_page()
    pdf.section_title("Sentiment Analysis")
    pdf.kv_row("Overall Label", sentiment_summary.get("overall_label", "N/A"))
    pdf.kv_row("Overall Score", sentiment_summary.get("overall_score", "N/A"))
    pdf.kv_row("Confidence", sentiment_summary.get("confidence", "N/A"))
    pdf.kv_row("Sentence Count", sentiment_summary.get("sentence_count", "N/A"))

    distribution = sentiment_summary.get("distribution", {})
    if distribution:
        pdf.kv_row("Positive %", distribution.get("positive_pct", "N/A"))
        pdf.kv_row("Neutral %", distribution.get("neutral_pct", "N/A"))
        pdf.kv_row("Negative %", distribution.get("negative_pct", "N/A"))

    if charts:
        if charts.get("sentiment_timeline") is not None:
            pdf.add_plotly_figure(charts["sentiment_timeline"], "Sentiment Over Time")
        if charts.get("sentiment_distribution") is not None:
            pdf.add_plotly_figure(charts["sentiment_distribution"], "Sentiment Distribution")

    positives = sentiment_summary.get("most_positive", [])
    negatives = sentiment_summary.get("most_negative", [])

    if positives:
        pdf.section_title("Top Positive Moments")
        for item in positives:
            pdf.body_text(
                f"[{item.get('start', 0):.1f}s - {item.get('end', 0):.1f}s] "
                f"{item.get('text', '')} (score={item.get('score', '')})"
            )

    if negatives:
        pdf.section_title("Top Negative Moments")
        for item in negatives:
            pdf.body_text(
                f"[{item.get('start', 0):.1f}s - {item.get('end', 0):.1f}s] "
                f"{item.get('text', '')} (score={item.get('score', '')})"
            )

    # ---------------- TONE SECTION ----------------
    pdf.add_page()
    pdf.section_title("Tone Analysis")
    pdf.kv_row("Dominant Tone", tone_summary.get("dominant_tone", "N/A"))
    pdf.kv_row("Tone Score", tone_summary.get("tone_score", "N/A"))
    pdf.kv_row("Confidence", tone_summary.get("confidence", "N/A"))

    tone_dist = tone_summary.get("distribution", {})
    if tone_dist:
        pdf.section_title("Tone Distribution")
        for label, value in tone_dist.items():
            pdf.kv_row(label.title(), f"{value}%")

    if charts and charts.get("tone_distribution") is not None:
        pdf.add_plotly_figure(charts["tone_distribution"], "Tone Distribution Chart")

    # ---------------- BIAS SECTION ----------------
    pdf.add_page()
    pdf.section_title("Bias Detection")
    pdf.kv_row("Bias Score", bias_summary.get("bias_score", "N/A"))
    pdf.kv_row("Bias Level", bias_summary.get("bias_level", "N/A"))
    pdf.kv_row("Total Flags", bias_summary.get("total_flags", "N/A"))

    category_distribution = analysis_data.get("bias", {}).get("category_distribution", {})
    if category_distribution:
        pdf.section_title("Bias Category Distribution")
        for cat, val in category_distribution.items():
            pdf.kv_row(cat, f"{val}%")

    if charts:
        if charts.get("bias_timeline") is not None:
            pdf.add_plotly_figure(charts["bias_timeline"], "Bias Flags Over Time")
        if charts.get("bias_distribution") is not None:
            pdf.add_plotly_figure(charts["bias_distribution"], "Bias Category Distribution Chart")

    flagged_instances = analysis_data.get("bias", {}).get("flagged_instances", [])
    if flagged_instances:
        pdf.section_title("Flagged Bias Instances")
        for idx, item in enumerate(flagged_instances[:10], start=1):
            match_str = ", ".join(
                [f"{m.get('keyword')} [{m.get('category')}]" for m in item.get("matches", [])]
            )
            text = (
                f"{idx}. [{item.get('start', 0):.1f}s - {item.get('end', 0):.1f}s] "
                f"Sentence: {item.get('sentence', '')}\n"
                f"Matches: {match_str}\n"
                f"Final Score: {item.get('final_score', '')} | Level: {item.get('level', '')}"
            )
            pdf.body_text(text)

    # ---------------- EMOTIONPRINT SECTION ----------------
    pdf.add_page()
    pdf.section_title("EmotionPrint Analysis")
    pdf.kv_row("Authenticity", emotion_summary.get("authenticity_pct", "N/A"))
    pdf.kv_row("Mismatch Count", emotion_summary.get("mismatch_count", "N/A"))
    pdf.kv_row("Sarcasm Count", emotion_summary.get("sarcasm_count", "N/A"))
    pdf.kv_row("Suppression Count", emotion_summary.get("suppression_count", "N/A"))
    pdf.kv_row("Irony Count", emotion_summary.get("irony_count", "N/A"))

    if charts:
        if charts.get("emotion_timeline") is not None:
            pdf.add_plotly_figure(charts["emotion_timeline"], "EmotionPrint Divergence Over Time")
        if charts.get("emotion_summary") is not None:
            pdf.add_plotly_figure(charts["emotion_summary"], "EmotionPrint Summary")

    flagged_moments = analysis_data.get("emotionprint", {}).get("flagged_moments", [])
    if flagged_moments:
        pdf.section_title("Flagged Emotional Mismatches")
        for idx, item in enumerate(flagged_moments[:10], start=1):
            text = (
                f"{idx}. [{item.get('time_label', 'N/A')}] "
                f"{item.get('type', 'Mismatch')} | "
                f"Confidence: {item.get('confidence', 'N/A')} | "
                f"Divergence: {item.get('divergence', 'N/A')}"
            )
            pdf.body_text(text)

    # ---------------- TRANSCRIPT SECTION ----------------
    pdf.add_page()
    pdf.section_title("Transcript")
    transcript_text = analysis_data.get("transcript_text", "")
    if transcript_text:
        pdf.body_text(transcript_text[:12000])  # limit for PDF size
    else:
        pdf.body_text("Transcript not available.")

    pdf_output = pdf.output(dest="S")

    if isinstance(pdf_output, bytearray):
        return bytes(pdf_output)
    elif isinstance(pdf_output, str):
        return pdf_output.encode("latin-1", errors="replace")
    return pdf_output
