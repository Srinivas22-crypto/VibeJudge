import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List


def create_sentiment_timeline(timeline_data: list[dict]) -> go.Figure:
    """
    Create sentiment timeline chart
    
    Args:
        timeline_data: List of timeline bins with sentiment scores
    
    Returns:
        Plotly figure object
    """
    if not timeline_data:
        return _empty_figure("No timeline data available")
    
    time_labels = [bin["time_label"] for bin in timeline_data]
    sentiment_scores = [bin["avg_sentiment"] for bin in timeline_data]
    
    # Color mapping
    colors = [
        '#2ecc71' if score > 0.2 else  # Green for positive
        '#e74c3c' if score < -0.2 else  # Red for negative
        '#95a5a6'  # Gray for neutral
        for score in sentiment_scores
    ]
    
    fig = go.Figure()
    
    # Line chart
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=sentiment_scores,
        mode='lines+markers',
        name='Sentiment',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8, color=colors),
        hovertemplate='<b>Time:</b> %{x}<br>' +
                      '<b>Sentiment:</b> %{y:.2f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Positive/Negative zones
    fig.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.1, line_width=0)
    
    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig


def create_sentiment_distribution_pie(sentiment_results: dict) -> go.Figure:
    """
    Create pie chart for sentiment distribution
    
    Args:
        sentiment_results: Sentiment analysis results
    
    Returns:
        Plotly figure object
    """
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        sentiment_results['positive_ratio'] * 100,
        sentiment_results['neutral_ratio'] * 100,
        sentiment_results['negative_ratio'] * 100
    ]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Overall Sentiment Distribution",
        height=400,
        showlegend=True
    )
    
    return fig


def create_tone_heatmap(tone_timeline: list[dict]) -> go.Figure:
    """
    Create heatmap showing tone changes over time
    
    Args:
        tone_timeline: Tone timeline data
    
    Returns:
        Plotly figure object
    """
    if not tone_timeline:
        return _empty_figure("No tone timeline data available")
    
    # Extract data
    time_labels = [bin["time_label"] for bin in tone_timeline]
    tones = [bin["dominant_tone"] for bin in tone_timeline]
    
    # Tone to numeric mapping
    tone_map = {
        "calm": 0,
        "confident": 1,
        "persuasive": 2,
        "excited": 3,
        "anxious": 4,
        "aggressive": 5
    }
    
    tone_values = [tone_map.get(t, 0) for t in tones]
    
    # Color scale
    colorscale = [
        [0, '#3498db'],     # Calm - Blue
        [0.2, '#2ecc71'],   # Confident - Green
        [0.4, '#f39c12'],   # Persuasive - Orange
        [0.6, '#9b59b6'],   # Excited - Purple
        [0.8, '#e67e22'],   # Anxious - Dark Orange
        [1, '#e74c3c']      # Aggressive - Red
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=[tone_values],
        x=time_labels,
        y=['Tone'],
        colorscale=colorscale,
        showscale=False,
        hovertemplate='<b>Time:</b> %{x}<br>' +
                      '<b>Tone:</b> ' + 
                      np.array(tones)[np.newaxis, :].tolist()[0] +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Tone Heatmap Over Time",
        xaxis_title="Time",
        height=200,
        yaxis=dict(showticklabels=False)
    )
    
    return fig


def create_combined_dashboard(
    sentiment_results: dict,
    tone_results: dict
) -> go.Figure:
    """
    Create comprehensive dashboard with multiple subplots
    
    Args:
        sentiment_results: Sentiment analysis results
        tone_results: Tone analysis results
    
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sentiment Timeline',
            'Sentiment Distribution',
            'Tone Distribution',
            'Key Metrics'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'pie'}],
            [{'type': 'bar'}, {'type': 'indicator'}]
        ],
        row_heights=[0.6, 0.4]
    )
    
    # 1. Sentiment Timeline (row 1, col 1)
    if sentiment_results.get("timeline"):
        timeline = sentiment_results["timeline"]
        time_labels = [bin["time_label"] for bin in timeline]
        sentiment_scores = [bin["avg_sentiment"] for bin in timeline]
        
        fig.add_trace(
            go.Scatter(
                x=time_labels,
                y=sentiment_scores,
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='#3498db', width=2)
            ),
            row=1, col=1
        )
    
    # 2. Sentiment Pie (row 1, col 2)
    sentiment_labels = ['Positive', 'Neutral', 'Negative']
    sentiment_values = [
        sentiment_results['positive_ratio'] * 100,
        sentiment_results['neutral_ratio'] * 100,
        sentiment_results['negative_ratio'] * 100
    ]
    sentiment_colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    fig.add_trace(
        go.Pie(
            labels=sentiment_labels,
            values=sentiment_values,
            marker=dict(colors=sentiment_colors),
            hole=0.3,
            name='Sentiment'
        ),
        row=1, col=2
    )
    
    # 3. Tone Bar Chart (row 2, col 1)
    if tone_results.get("tone_distribution"):
        tone_dist = tone_results["tone_distribution"]
        tone_labels = list(tone_dist.keys())
        tone_values = [tone_dist[t] * 100 for t in tone_labels]
        
        fig.add_trace(
            go.Bar(
                x=tone_labels,
                y=tone_values,
                marker_color='#9b59b6',
                name='Tone'
            ),
            row=2, col=1
        )
    
    # 4. Key Metrics Indicator (row 2, col 2)
    overall_score = sentiment_results['overall_score']
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            title={'text': "Overall Sentiment"},
            delta={'reference': 0},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0
                }
            }
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="VibeJudge Analysis Dashboard",
        title_font_size=20
    )
    
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Create empty figure with message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=400
    )
    return fig
    
def create_bias_timeline(bias_timeline: list[dict]) -> go.Figure:
    """
    Bar chart showing bias flag count per 30-second bin.

    Args:
        bias_timeline: From BiasDetector.analyze_text()["timeline"]

    Returns:
        Plotly figure
    """
    if not bias_timeline:
        return _empty_figure("No bias timeline data available")

    time_labels  = [b["time_label"]  for b in bias_timeline]
    bias_counts  = [b["bias_count"]  for b in bias_timeline]

    # Color by intensity
    max_count = max(bias_counts) if max(bias_counts) > 0 else 1
    bar_colors = [
        f"rgba(231, 76, 60, {0.3 + 0.7 * (c / max_count)})"
        for c in bias_counts
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=time_labels,
            y=bias_counts,
            marker_color=bar_colors,
            hovertemplate="<b>Time:</b> %{x}<br><b>Bias Flags:</b> %{y}<extra></extra>"
        )
    ])

    fig.update_layout(
        title="Bias Flags Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Number of Bias Flags",
        template="plotly_white",
        height=350
    )

    return fig


def create_bias_category_chart(category_dist: dict) -> go.Figure:
    """
    Horizontal bar chart of bias categories.

    Args:
        category_dist: {category: fraction} from BiasDetector

    Returns:
        Plotly figure
    """
    if not category_dist:
        return _empty_figure("No bias categories detected")

    categories = [c.replace("_", " ").title() for c in category_dist.keys()]
    values     = [v * 100 for v in category_dist.values()]

    category_colors = {
        "Political Left":   "#3498db",
        "Political Right":  "#e74c3c",
        "Gender Bias":      "#9b59b6",
        "Loaded Language":  "#e67e22",
        "Weasel Words":     "#95a5a6"
    }

    colors = [category_colors.get(c, "#34495e") for c in categories]

    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"
        )
    ])

    fig.update_layout(
        title="Bias Category Distribution",
        xaxis_title="Percentage of Bias Flags",
        template="plotly_white",
        height=350,
        margin=dict(l=160)
    )

    return fig


def create_emotionprint_timeline(ep_results: dict) -> go.Figure:
    """
    Scatter/line plot of divergence scores over podcast time.

    Args:
        ep_results: From EmotionPrintAnalyzer.analyze_full_transcript()

    Returns:
        Plotly figure
    """
    segments = ep_results.get("all_segments", [])

    if not segments:
        return _empty_figure("No EmotionPrint™ data available")

    timestamps  = [s["start_time"]       for s in segments]
    divergences = [s["divergence_score"] for s in segments]
    states      = [s["emotional_state"]  for s in segments]

    state_colors = {
        "Authentic":            "#2ecc71",
        "Sarcasm":              "#e74c3c",
        "Irony":                "#f39c12",
        "Emotional Suppression":"#9b59b6",
        "Emotional Mismatch":   "#e67e22"
    }

    point_colors = [state_colors.get(st, "#95a5a6") for st in states]
    time_labels  = [
        f"{int(t//60):02d}:{int(t%60):02d}" for t in timestamps
    ]

    fig = go.Figure()

    # Divergence line
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=divergences,
        mode="lines+markers",
        name="Divergence Score",
        line=dict(color="#3498db", width=2),
        marker=dict(size=8, color=point_colors),
        customdata=states,
        hovertemplate=(
            "<b>Time:</b> %{x}<br>"
            "<b>Divergence:</b> %{y:.2f}<br>"
            "<b>State:</b> %{customdata}<extra></extra>"
        )
    ))

    # Threshold line
    fig.add_hline(
        y=0.55,
        line_dash="dash",
        line_color="red",
        opacity=0.6,
        annotation_text="Divergence Threshold",
        annotation_position="right"
    )

    fig.update_layout(
        title="EmotionPrint™ Divergence Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Divergence Score (0–1)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=380
    )

    return fig


def create_emotionprint_summary_chart(ep_results: dict) -> go.Figure:
    """
    Donut chart summarizing emotional states from EmotionPrint™.

    Args:
        ep_results: Full EmotionPrint™ analysis result

    Returns:
        Plotly figure
    """
    labels = ["Authentic", "Sarcasm", "Irony", "Suppression", "Mismatch"]
    total  = ep_results.get("total_segments_analyzed", 1)
    flagged = ep_results.get("flagged_segments_count", 0)

    values = [
        total - flagged,
        ep_results.get("sarcasm_instances", 0),
        ep_results.get("irony_instances", 0),
        ep_results.get("suppression_instances", 0),
        max(0, flagged - (
            ep_results.get("sarcasm_instances", 0) +
            ep_results.get("irony_instances", 0) +
            ep_results.get("suppression_instances", 0)
        ))
    ]

    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#e67e22"]

    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.45,
            textinfo="label+percent",
            textposition="outside"
        )
    ])

    auth_score = ep_results.get("authenticity_score", 100.0)

    fig.update_layout(
        title=f"EmotionPrint™ Summary — Authenticity: {auth_score:.0f}%",
        height=420,
        showlegend=True
    )

    return fig


def generate_color_coded_transcript(
    sentiment_sentences: list[dict],
    bias_flags: list[dict]
) -> str:
    """
    Generate full color-coded HTML transcript.

    Color scheme:
    - Green background  → Positive sentiment
    - Red background    → Negative sentiment
    - Orange underline  → Bias keyword detected
    - Purple border     → Sarcasm / EmotionPrint™ flag

    Args:
        sentiment_sentences: From SentimentAnalyzer
        bias_flags         : From BiasDetector

    Returns:
        HTML string ready for st.markdown(..., unsafe_allow_html=True)
    """
    # Build bias keyword lookup
    bias_keywords = {
        flag["keyword"].lower(): flag
        for flag in bias_flags
    }

    html_parts = ["<div style='font-family: Georgia, serif; line-height: 2.2;'>"]

    for sentence_data in sentiment_sentences:
        text  = sentence_data.get("text", "")
        label = sentence_data.get("label", "neutral")
        score = abs(sentence_data.get("score", 0))

        # Background color for sentiment
        if label == "positive":
            bg = f"rgba(46, 204, 113, {max(0.08, score * 0.25)})"
        elif label == "negative":
            bg = f"rgba(231, 76, 60, {max(0.08, score * 0.25)})"
        else:
            bg = "transparent"

        # Check for bias keywords
        display_text = text
        for keyword in bias_keywords:
            if keyword in text.lower():
                # Replace with orange underlined version
                import re
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                display_text = pattern.sub(
                    f"<span style='"
                    f"text-decoration: underline wavy #e67e22; "
                    f"color: #c0392b; font-weight: bold;"
                    f"' title='Bias: {bias_keywords[keyword]['category']}'>"
                    f"{keyword}"
                    f"</span>",
                    display_text
                )
                break  # One highlight per sentence is enough for readability

        # Sentiment label badge
        badge_color = {
            "positive": "#27ae60",
            "negative": "#c0392b",
            "neutral":  "#7f8c8d"
        }.get(label, "#7f8c8d")

        badge = (
            f"<sup style='"
            f"background:{badge_color}; color:white; "
            f"font-size:9px; padding:1px 4px; border-radius:8px; "
            f"margin-left:4px; font-family:monospace;"
            f"'>{label[:3].upper()}</sup>"
        )

        html_parts.append(
            f"<span style='"
            f"background-color:{bg}; "
            f"padding:3px 6px; border-radius:4px; "
            f"display:inline; margin:2px 0;"
            f"'>{display_text}{badge}</span> "
        )

    html_parts.append("</div>")
    return "".join(html_parts)
