import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List


def create_sentiment_timeline(timeline_data: List[Dict]) -> go.Figure:
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


def create_sentiment_distribution_pie(sentiment_results: Dict) -> go.Figure:
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


def create_tone_heatmap(tone_timeline: List[Dict]) -> go.Figure:
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
    sentiment_results: Dict,
    tone_results: Dict
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
