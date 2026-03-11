import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, HRFlowable, KeepTogether
)

logger = logging.getLogger(__name__)

# ─── Brand colors ──────────────────────────────────────────────
C_PRIMARY   = colors.HexColor("#2c3e50")
C_BLUE      = colors.HexColor("#3498db")
C_GREEN     = colors.HexColor("#27ae60")
C_RED       = colors.HexColor("#e74c3c")
C_ORANGE    = colors.HexColor("#e67e22")
C_PURPLE    = colors.HexColor("#9b59b6")
C_GRAY      = colors.HexColor("#95a5a6")
C_LIGHT     = colors.HexColor("#ecf0f1")
C_WHITE     = colors.white


def _styles():
    """Return custom paragraph styles"""
    base = getSampleStyleSheet()

    custom = {
        "Title": ParagraphStyle(
            "VJTitle",
            parent=base["Title"],
            fontSize=26, textColor=C_PRIMARY,
            spaceAfter=6, alignment=TA_CENTER, fontName="Helvetica-Bold"
        ),
        "Subtitle": ParagraphStyle(
            "VJSubtitle",
            parent=base["Normal"],
            fontSize=11, textColor=C_GRAY,
            spaceAfter=20, alignment=TA_CENTER
        ),
        "H2": ParagraphStyle(
            "VJH2",
            parent=base["Heading2"],
            fontSize=15, textColor=C_BLUE,
            spaceBefore=18, spaceAfter=8,
            fontName="Helvetica-Bold",
            borderPad=4
        ),
        "H3": ParagraphStyle(
            "VJH3",
            parent=base["Heading3"],
            fontSize=12, textColor=C_PRIMARY,
            spaceBefore=10, spaceAfter=6,
            fontName="Helvetica-Bold"
        ),
        "Body": ParagraphStyle(
            "VJBody",
            parent=base["Normal"],
            fontSize=10, leading=15,
            textColor=colors.HexColor("#2d3436"),
            spaceAfter=6
        ),
        "Quote": ParagraphStyle(
            "VJQuote",
            parent=base["Normal"],
            fontSize=10, leading=14,
            textColor=colors.HexColor("#636e72"),
            leftIndent=20, rightIndent=20,
            borderPad=6, spaceAfter=8
        ),
        "Caption": ParagraphStyle(
            "VJCaption",
            parent=base["Normal"],
            fontSize=8, textColor=C_GRAY,
            alignment=TA_CENTER, spaceAfter=6
        )
    }
    return custom


# ─── Chart generators (Matplotlib → BytesIO → ReportLab Image) ─

def _chart_buffer(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def _sentiment_timeline_chart(sentiment: Dict) -> Optional[io.BytesIO]:
    timeline = sentiment.get("timeline", [])
    if not timeline:
        return None

    labels  = [b.get("time_label","") for b in timeline]
    scores  = [b.get("avg_sentiment", 0) for b in timeline]

    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    ax.fill_between(range(len(scores)), scores, 0,
                    where=[s >= 0 for s in scores],
                    alpha=0.3, color="#27ae60", label="Positive")
    ax.fill_between(range(len(scores)), scores, 0,
                    where=[s < 0 for s in scores],
                    alpha=0.3, color="#e74c3c", label="Negative")
    ax.plot(range(len(scores)), scores,
            color="#2c3e50", linewidth=2, marker="o", markersize=4)
    ax.axhline(0, color="#636e72", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=7)
    ax.set_ylabel("Sentiment Score", fontsize=8)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Sentiment Timeline", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _chart_buffer(fig)


def _sentiment_pie_chart(sentiment: Dict) -> io.BytesIO:
    sizes  = [
        sentiment.get("positive_ratio", 0) * 100,
        sentiment.get("neutral_ratio",  0) * 100,
        sentiment.get("negative_ratio", 0) * 100,
    ]
    labels = ["Positive", "Neutral", "Negative"]
    colors_list = ["#27ae60", "#95a5a6", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    wedges, _, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_list,
        autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.6)   # donut
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title("Sentiment Distribution", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _chart_buffer(fig)


def _tone_bar_chart(tone: Dict) -> Optional[io.BytesIO]:
    dist = tone.get("tone_distribution", {})
    if not dist:
        return None

    tones  = list(dist.keys())
    values = [dist[t] * 100 for t in tones]
    palette = ["#3498db","#e74c3c","#f39c12","#9b59b6","#27ae60","#1abc9c"]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    bars = ax.bar(tones, values,
                  color=palette[:len(tones)], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylabel("Percentage (%)", fontsize=8)
    ax.set_title("Tone Distribution", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25 + 5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _chart_buffer(fig)


def _bias_chart(bias: Dict) -> Optional[io.BytesIO]:
    cat_dist = bias.get("category_distribution", {})
    if not cat_dist:
        return None

    cats   = [c.replace("_"," ").title() for c in cat_dist.keys()]
    values = [v * 100 for v in cat_dist.values()]
    palette = ["#3498db","#e74c3c","#9b59b6","#e67e22","#95a5a6"]

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    bars = ax.barh(cats, values,
                   color=palette[:len(cats)], edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=7.5)
    ax.set_xlabel("Percentage (%)", fontsize=8)
    ax.set_title("Bias Category Distribution", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.3 + 5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _chart_buffer(fig)


def _emotionprint_gauge(score: float) -> io.BytesIO:
    """Circular gauge for authenticity score"""
    fig, ax = plt.subplots(figsize=(3.0, 2.8),
                            subplot_kw=dict(polar=True))

    theta_range = np.linspace(0, np.pi, 200)
    # Background arc
    ax.plot(theta_range, [1]*200, color="#ecf0f1", linewidth=12)
    # Value arc
    filled = np.linspace(0, np.pi * (score/100), 200)
    color = "#27ae60" if score >= 70 else "#e67e22" if score >= 40 else "#e74c3c"
    ax.plot(filled, [1]*len(filled), color=color, linewidth=12)

    ax.set_ylim(0, 1.4)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    ax.axis("off")
    ax.text(0, 0, f"{score:.0f}%",
            ha="center", va="center",
            fontsize=20, fontweight="bold", color=color)
    ax.text(0, -0.35, "Authenticity",
            ha="center", va="center", fontsize=9, color="#636e72")
    fig.tight_layout()
    return _chart_buffer(fig)


# ─── Table helper ────────────────────────────────────────────────

def _metric_table(data: list, header_color=None) -> Table:
    """Build a styled 2-column metric table"""
    if header_color is None:
        header_color = C_BLUE

    t = Table(data, colWidths=[2.8*inch, 3.8*inch])
    style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  header_color),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("BACKGROUND",   (0, 1), (0, -1),  C_LIGHT),
        ("FONTNAME",     (0, 1), (0, -1),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#dfe6e9")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_LIGHT])
    ])
    t.setStyle(style)
    return t


# ─── Main generator ─────────────────────────────────────────────

def generate_pdf_report(
    podcast_id: str,
    filename: str,
    transcript_data: Dict,
    sentiment_results: Dict,
    tone_results: Dict,
    output_path: str,
    bias_results: Optional[Dict] = None,
    emotionprint_results: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive 7-section PDF report.

    Sections:
    1. Cover Page
    2. Executive Summary
    3. Sentiment Analysis
    4. Tone Analysis
    5. Bias Detection
    6. EmotionPrint™ (if available)
    7. Transcript Excerpt
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch,   bottomMargin=0.75*inch
    )

    st  = _styles()
    els = []   # Elements list

    duration_min = transcript_data.get("duration", 0) / 60
    word_count   = transcript_data.get("word_count", "N/A")
    language     = transcript_data.get("language", "en").upper()

    # ════════════════════════════════════════════════════
    # SECTION 1: COVER PAGE
    # ════════════════════════════════════════════════════
    els.append(Spacer(1, 1.0*inch))
    els.append(Paragraph("🎙️ VibeJudge", st["Title"]))
    els.append(Paragraph("Podcast Intelligence Analysis Report", st["Subtitle"]))
    els.append(HRFlowable(width="100%", thickness=2,
                           color=C_BLUE, spaceAfter=20))
    els.append(Spacer(1, 0.3*inch))

    cover_data = [
        ["Field", "Value"],
        ["Podcast File",    filename],
        ["Analysis ID",     podcast_id],
        ["Duration",        f"{duration_min:.1f} minutes"],
        ["Word Count",      str(word_count)],
        ["Language",        language],
        ["Analysis Date",   datetime.now().strftime("%B %d, %Y at %H:%M")],
        ["VibeJudge Ver.",  "v1.4 (Week 4)"]
    ]
    els.append(_metric_table(cover_data))
    els.append(Spacer(1, 0.5*inch))

    # Overall verdict banner
    overall_sent  = sentiment_results.get("overall_sentiment","neutral").upper()
    overall_tone  = tone_results.get("dominant_tone","calm").upper()
    bias_level    = (bias_results or {}).get("bias_level","N/A")
    auth_score    = (emotionprint_results or {}).get("authenticity_score", 100.0)

    verdict_data  = [
        ["SENTIMENT", "DOMINANT TONE", "BIAS LEVEL", "AUTHENTICITY"],
        [overall_sent, overall_tone, bias_level, f"{auth_score:.0f}%"]
    ]
    verdict_table = Table(verdict_data, colWidths=[1.6*inch]*4)
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), C_PRIMARY),
        ("TEXTCOLOR",    (0,0),(-1,0), C_WHITE),
        ("FONTNAME",     (0,0),(-1,-1),"Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,0), 9),
        ("FONTSIZE",     (0,1),(-1,1), 14),
        ("ALIGN",        (0,0),(-1,-1),"CENTER"),
        ("VALIGN",       (0,0),(-1,-1),"MIDDLE"),
        ("BOTTOMPADDING",(0,0),(-1,-1),12),
        ("TOPPADDING",   (0,0),(-1,-1),12),
        ("GRID",         (0,0),(-1,-1),0.5, C_GRAY),
        ("ROWBACKGROUNDS",(0,1),(-1,1),[C_LIGHT])
    ]))
    els.append(verdict_table)
    els.append(PageBreak())

    # ════════════════════════════════════════════════════
    # SECTION 2: EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════
    els.append(Paragraph("Executive Summary", st["H2"]))
    els.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=10))

    bias_count = (bias_results or {}).get("bias_flags_count", 0)
    bias_score = (bias_results or {}).get("overall_bias_score", 0)
    sarc_count = (emotionprint_results or {}).get("sarcasm_instances", 0)

    summary_text = (
        f"This podcast episode — <b>{filename}</b> — was analyzed using the "
        f"VibeJudge multimodal AI pipeline. The {duration_min:.1f}-minute recording "
        f"containing {word_count} words was processed through five analytical stages: "
        f"automatic speech recognition, sentiment classification, tone detection, "
        f"bias detection, and prosody-semantic divergence analysis (EmotionPrint™).<br/><br/>"
        f"<b>Key Findings:</b> The overall content exhibits "
        f"<b>{sentiment_results.get('overall_sentiment','neutral')}</b> sentiment "
        f"(score: {sentiment_results.get('overall_score',0):+.2f}) with a predominantly "
        f"<b>{tone_results.get('dominant_tone','calm')}</b> tone. "
        f"Bias analysis identified <b>{bias_count} flagged instance(s)</b> "
        f"with an overall bias score of <b>{bias_score:.0f}/100</b> "
        f"({bias_level} bias level). "
        f"EmotionPrint™ scored content authenticity at <b>{auth_score:.0f}%</b>, "
        f"detecting {sarc_count} potential sarcasm instance(s)."
    )
    els.append(Paragraph(summary_text, st["Body"]))
    els.append(Spacer(1, 0.2*inch))

    # Summary metrics table
    pos_r = sentiment_results.get("positive_ratio",0)
    neu_r = sentiment_results.get("neutral_ratio",0)
    neg_r = sentiment_results.get("negative_ratio",0)

    summary_metrics = [
        ["Metric", "Value"],
        ["Overall Sentiment",       f"{sentiment_results.get('overall_sentiment','N/A').capitalize()} ({sentiment_results.get('overall_score',0):+.3f})"],
        ["Sentiment Confidence",    f"{sentiment_results.get('confidence',0)*100:.1f}%"],
        ["Positive / Neutral / Negative", f"{pos_r*100:.1f}% / {neu_r*100:.1f}% / {neg_r*100:.1f}%"],
        ["Dominant Tone",           tone_results.get('dominant_tone','N/A').capitalize()],
        ["Tone Confidence",         f"{tone_results.get('confidence',0)*100:.1f}%"],
        ["Bias Score",              f"{bias_score:.0f} / 100  ({bias_level})"],
        ["Bias Flags",              str(bias_count)],
        ["Authenticity Score",      f"{auth_score:.0f}%"],
        ["Sarcasm Detected",        str(sarc_count)],
    ]
    els.append(_metric_table(summary_metrics))
    els.append(PageBreak())

    # ════════════════════════════════════════════════════
    # SECTION 3: SENTIMENT ANALYSIS
    # ════════════════════════════════════════════════════
    els.append(Paragraph("Sentiment Analysis", st["H2"]))
    els.append(HRFlowable(width="100%", thickness=1, color=C_BLUE, spaceAfter=10))

    sent_detail = [
        ["Metric", "Value"],
        ["Overall Sentiment", sentiment_results.get("overall_sentiment","N/A").capitalize()],
        ["Overall Score",     f"{sentiment_results.get('overall_score',0):+.3f}"],
        ["Confidence",        f"{sentiment_results.get('confidence',0)*100:.1f}%"],
        ["Sentences Analyzed",str(sentiment_results.get("sentence_count",0))],
        ["Positive Ratio",    f"{sentiment_results.get('positive_ratio',0)*100:.1f}%"],
        ["Neutral Ratio",     f"{sentiment_results.get('neutral_ratio',0)*100:.1f}%"],
        ["Negative Ratio",    f"{sentiment_results.get('negative_ratio',0)*100:.1f}%"],
    ]
    els.append(_metric_table(sent_detail, header_color=C_GREEN))
    els.append(Spacer(1, 0.2*inch))

    # Timeline + Pie side by side
    timeline_buf = _sentiment_timeline_chart(sentiment_results)
    pie_buf      = _sentiment_pie_chart(sentiment_results)

    if timeline_buf and pie_buf:
        chart_row = Table(
            [[RLImage(timeline_buf, width=4.2*inch, height=2.0*inch),
              RLImage(pie_buf,      width=2.6*inch, height=2.0*inch)]],
            colWidths=[4.3*inch, 2.7*inch]
        )
        els.append(chart_row)
        els.append(Paragraph("Left: Sentiment timeline (30-sec bins). "
                              "Right: Overall sentiment distribution.", st["Caption"]))

    # Key moments
    km = sentiment_results.get("key_moments",{})
    els.append(Spacer(1, 0.15*inch))
    els.append(Paragraph("Key Moments", st["H3"]))

    if km.get("most_positive"):
        mp = km["most_positive"]
        els.append(Paragraph(
            f"<b>Most Positive Sentence:</b><br/>"
            f"<i>\"{mp['text'][:200]}\"</i>  (Score: {mp['score']:+.2f})",
            st["Quote"]
        ))
    if km.get("most_negative"):
        mn = km["most_negative"]
        els.append(Paragraph(
            f"<b>Most Negative Sentence:</b><br/>"
            f"<i>\"{mn['text'][:200]}\"</i>  (Score: {mn['score']:+.2f})",
            st["Quote"]
        ))
    els.append(PageBreak())

    # ════════════════════════════════════════════════════
    # SECTION 4: TONE ANALYSIS
    # ════════════════════════════════════════════════════
    els.append(Paragraph("Tone Analysis", st["H2"]))
    els.append(HRFlowable(width="100%", thickness=1, color=C_PURPLE, spaceAfter=10))

    tone_detail = [
        ["Metric", "Value"],
        ["Dominant Tone",  tone_results.get("dominant_tone","N/A").capitalize()],
        ["Tone Score",     f"{tone_results.get('dominant_score',0):.3f}"],
        ["Confidence",     f"{tone_results.get('confidence',0)*100:.1f}%"],
        ["Sentences",      str(tone_results.get("sentence_count",0))],
    ]

    dist = tone_results.get("tone_distribution",{})
    for tone_name, pct in sorted(dist.items(), key=lambda x: -x[1]):
        tone_detail.append([
            f"  {tone_name.capitalize()} ratio",
            f"{pct*100:.1f}%"
        ])

    els.append(_metric_table(tone_detail, header_color=C_PURPLE))
    els.append(Spacer(1, 0.2*inch))

    tone_chart_buf = _tone_bar_chart(tone_results)
    if tone_chart_buf:
        els.append(RLImage(tone_chart_buf, width=5.5*inch, height=2.5*inch))
        els.append(Paragraph("Tone distribution across all sentences.", st["Caption"]))

    # Examples
    examples = tone_results.get("tone_examples",{})
    if examples:
        els.append(Spacer(1, 0.15*inch))
        els.append(Paragraph("Representative Tone Examples", st["H3"]))
        ex_data = [["Tone", "Example Text", "Score"]]
        for tone_name, ex in list(examples.items())[:5]:
            ex_data.append([
                tone_name.capitalize(),
                ex["text"][:80] + ("…" if len(ex["text"])>80 else ""),
                f"{ex['score']:.2f}"
            ])
        ex_table = Table(ex_data, colWidths=[1.2*inch, 4.8*inch, 0.7*inch])
        ex_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0),(-1,0), C_PURPLE),
            ("TEXTCOLOR",   (0,0),(-1,0), C_WHITE),
            ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0),(-1,-1), 8),
            ("ALIGN",       (2,0),(2,-1), "CENTER"),
            ("GRID",        (0,0),(-1,-1), 0.4, C_GRAY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_LIGHT]),
            ("LEFTPADDING", (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1),7),
        ]))
        els.append(ex_table)
    els.append(PageBreak())

    # ════════════════════════════════════════════════════
    # SECTION 5: BIAS DETECTION
    # ════════════════════════════════════════════════════
    els.append(Paragraph("Bias Detection", st["H2"]))
    els.append(HRFlowable(width="100%", thickness=1, color=C_RED, spaceAfter=10))

    if bias_results:
        bias_detail = [
            ["Metric", "Value"],
            ["Bias Score",    f"{bias_results.get('overall_bias_score',0):.1f} / 100"],
            ["Bias Level",    bias_results.get("bias_level","N/A")],
            ["Total Flags",   str(bias_results.get("bias_flags_count",0))],
        ]
        for cat, pct in (bias_results.get("category_distribution",{}) or {}).items():
            bias_detail.append([
                f"  {cat.replace('_',' ').title()}",
                f"{pct*100:.1f}% of flags"
            ])
        els.append(_metric_table(bias_detail, header_color=C_RED))
        els.append(Spacer(1, 0.2*inch))

        bias_chart_buf = _bias_chart(bias_results)
        if bias_chart_buf:
            els.append(RLImage(bias_chart_buf, width=5.0*inch, height=2.3*inch))
            els.append(Paragraph("Bias flags by category.", st["Caption"]))

        # Top flagged instances
        flags = bias_results.get("bias_flags",[])
        if flags:
            els.append(Spacer(1, 0.15*inch))
            els.append(Paragraph("Top Flagged Instances", st["H3"]))
            flag_data = [["#","Keyword","Category","Severity","Timestamp"]]
            for i, fl in enumerate(flags[:8], 1):
                flag_data.append([
                    str(i),
                    fl.get("keyword","")[:30],
                    fl.get("category","").replace("_"," ").title(),
                    fl.get("severity",""),
                    fl.get("timestamp_formatted","N/A")
                ])
            flag_table = Table(flag_data,
                               colWidths=[0.3*inch,1.8*inch,1.8*inch,0.9*inch,0.9*inch])
            flag_table.setStyle(TableStyle([
                ("BACKGROUND",  (0,0),(-1,0), C_RED),
                ("TEXTCOLOR",   (0,0),(-1,0), C_WHITE),
                ("FONTNAME",    (0,0),(-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0),(-1,-1), 8),
                ("ALIGN",       (0,0),(0,-1), "CENTER"),
                ("GRID",        (0,0),(-1,-1), 0.4, C_GRAY),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_WHITE, C_LIGHT]),
                ("BOTTOMPADDING",(0,0),(-1,-1),7),
                ("LEFTPADDING", (0,0),(-1,-1), 7),
            ]))
            els.append(flag_table)
    else:
        els.append(Paragraph("Bias detection was not run for this analysis.", st["Body"]))

    els.append(PageBreak())

    # ════════════════════════════════════════════════════
    # SECTION 6: EMOTIONPRINT™
    # ════════════════════════════════════════════════════
    els.append(Paragraph("EmotionPrint™ — Prosody-Semantic Divergence", st["H2"]))
    els.append(HRFlowable(width="100%", thickness=1, color=C_ORANGE, spaceAfter=10))

    if emotionprint_results:
        ep_detail = [
            ["Metric", "Value"],
            ["Authenticity Score",   f"{emotionprint_results.get('authenticity_score',100):.1f}%"],
            ["Segments Analyzed",    str(emotionprint_results.get("total_segments_analyzed",0))],
            ["Flagged Segments",     str(emotionprint_results.get("flagged_segments_count",0))],
            ["Sarcasm Detected",     str(emotionprint_results.get("sarcasm_instances",0))],
            ["Emotional Suppression",str(emotionprint_results.get("suppression_instances",0))],
            ["Irony Detected",       str(emotionprint_results.get("irony_instances",0))],
        ]
        els.append(_metric_table(ep_detail, header_color=C_ORANGE))
        els.append(Spacer(1, 0.2*inch))

        auth_score_val = emotionprint_results.get("authenticity_score", 100.0)
        gauge_buf = _emotionprint_gauge(auth_score_val)
        els.append(RLImage(gauge_buf, width=2.8*inch, height=2.5*inch))
        els.append(Paragraph("Authenticity gauge (green ≥70%, orange 40-70%, red <40%).",
                              st["Caption"]))

        # Key sarcasm moment
        km_ep = emotionprint_results.get("key_moments",{})
        if km_ep.get("highest_sarcasm"):
            hs = km_ep["highest_sarcasm"]
            els.append(Spacer(1, 0.1*inch))
            els.append(Paragraph("Highest-Confidence Sarcasm Instance", st["H3"]))
            els.append(Paragraph(
                f"<b>Timestamp:</b> {hs.get('timestamp','N/A')}  |  "
                f"<b>Confidence:</b> {hs.get('confidence',0)*100:.0f}%<br/>"
                f"<b>Text:</b> <i>\"{hs.get('text','')[:200]}\"</i><br/>"
                f"<b>Explanation:</b> {hs.get('explanation','')}",
                st["Quote"]
            ))
    else:
        els.append(Paragraph("EmotionPrint™ analysis was not run.", st["Body"]))

    els.append(PageBreak())

    # ════════════════════════════════════════════════════
    # SECTION 7: TRANSCRIPT EXCERPT
    # ════════════════════════════════════════════════════
    els.append(Paragraph("Transcript Excerpt", st["H2"]))
    els.append(HRFlowable(width="100%", thickness=1, color=C_GRAY, spaceAfter=10))

    full_text = transcript_data.get("text","")
    excerpt   = full_text[:3000] + ("…" if len(full_text) > 3000 else "")

    els.append(Paragraph(
        f"<i>Showing first 3,000 characters of {len(full_text):,} total characters.</i>",
        st["Caption"]
    ))
    els.append(Spacer(1, 0.1*inch))
    els.append(Paragraph(excerpt, st["Body"]))

    # Build
    doc.build(els)
    logger.info(f"✓ Enhanced PDF saved to {output_path}")
    return output_path
