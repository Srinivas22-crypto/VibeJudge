import json
from datetime import datetime
from pathlib import Path
from typing import Dict

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import io


def generate_pdf_report(
    podcast_id: str,
    filename: str,
    transcript_data: Dict,
    sentiment_results: Dict,
    tone_results: Dict,
    output_path: str
) -> str:
    """
    Generate comprehensive PDF report
    
    Args:
        podcast_id: Unique podcast identifier
        filename: Original podcast filename
        transcript_data: Transcription results
        sentiment_results: Sentiment analysis results
        tone_results: Tone analysis results
        output_path: Path to save PDF
    
    Returns:
        Path to generated PDF file
    """
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # 1. Title Page
    elements.append(Paragraph("VibeJudge Analysis Report", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Podcast info
    info_data = [
        ['Podcast ID:', podcast_id],
        ['Filename:', filename],
        ['Duration:', f"{transcript_data.get('duration', 0)/60:.1f} minutes"],
        ['Word Count:', str(transcript_data.get('word_count', 'N/A'))],
        ['Language:', transcript_data.get('language', 'en').upper()],
        ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M')]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    elements.append(info_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # 2. Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    
    summary_text = f"""
    This podcast exhibits <b>{sentiment_results['overall_sentiment']}</b> sentiment 
    (score: {sentiment_results['overall_score']:.2f}) with a dominant 
    <b>{tone_results['dominant_tone']}</b> tone. The analysis processed 
    {sentiment_results['sentence_count']} sentences with 
    {sentiment_results['positive_ratio']*100:.1f}% positive, 
    {sentiment_results['neutral_ratio']*100:.1f}% neutral, and 
    {sentiment_results['negative_ratio']*100:.1f}% negative content.
    """
    
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # 3. Sentiment Analysis Section
    elements.append(PageBreak())
    elements.append(Paragraph("Sentiment Analysis", heading_style))
    
    # Sentiment metrics table
    sentiment_data = [
        ['Metric', 'Value'],
        ['Overall Sentiment', sentiment_results['overall_sentiment'].capitalize()],
        ['Overall Score', f"{sentiment_results['overall_score']:.3f}"],
        ['Confidence', f"{sentiment_results['confidence']*100:.1f}%"],
        ['Positive Ratio', f"{sentiment_results['positive_ratio']*100:.1f}%"],
        ['Neutral Ratio', f"{sentiment_results['neutral_ratio']*100:.1f}%"],
        ['Negative Ratio', f"{sentiment_results['negative_ratio']*100:.1f}%"]
    ]
    
    sentiment_table = Table(sentiment_data, colWidths=[3*inch, 3*inch])
    sentiment_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(sentiment_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Key moments
    if sentiment_results.get('key_moments'):
        elements.append(Paragraph("Key Moments", styles['Heading3']))
        
        pos_moment = sentiment_results['key_moments'].get('most_positive')
        neg_moment = sentiment_results['key_moments'].get('most_negative')
        
        if pos_moment:
            elements.append(Paragraph(
                f"<b>Most Positive:</b> <i>\"{pos_moment['text']}\"</i> (Score: {pos_moment['score']:.2f})",
                styles['Normal']
            ))
            elements.append(Spacer(1, 0.1*inch))
        
        if neg_moment:
            elements.append(Paragraph(
                f"<b>Most Negative:</b> <i>\"{neg_moment['text']}\"</i> (Score: {neg_moment['score']:.2f})",
                styles['Normal']
            ))
            elements.append(Spacer(1, 0.3*inch))
    
    # 4. Tone Analysis Section
    elements.append(PageBreak())
    elements.append(Paragraph("Tone Analysis", heading_style))
    
    # Tone metrics
    tone_data = [
        ['Metric', 'Value'],
        ['Dominant Tone', tone_results['dominant_tone'].capitalize()],
        ['Dominant Score', f"{tone_results['dominant_score']:.3f}"],
        ['Confidence', f"{tone_results['confidence']*100:.1f}%"]
    ]
    
    tone_table = Table(tone_data, colWidths=[3*inch, 3*inch])
    tone_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(tone_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Tone distribution
    if tone_results.get('tone_distribution'):
        elements.append(Paragraph("Tone Distribution", styles['Heading3']))
        
        tone_dist_data = [['Tone', 'Percentage']]
        for tone, percentage in tone_results['tone_distribution'].items():
            tone_dist_data.append([tone.capitalize(), f"{percentage*100:.1f}%"])
        
        tone_dist_table = Table(tone_dist_data, colWidths=[3*inch, 3*inch])
        tone_dist_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(tone_dist_table)
    
    # 5. Transcript Excerpt
    elements.append(PageBreak())
    elements.append(Paragraph("Transcript Excerpt", heading_style))
    
    excerpt_text = transcript_data.get('text', '')[:2000]  # First 2000 chars
    if len(transcript_data.get('text', '')) > 2000:
        excerpt_text += "..."
    
    elements.append(Paragraph(excerpt_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    return output_path
