import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def load_analysis_results(podcast_id: str, results_dir: str) -> Optional[Dict]:
    """Load all analysis JSON files for a given podcast_id"""
    rd = Path(results_dir)
    combined = {"podcast_id": podcast_id}

    for module in ["sentiment","tone","bias","emotionprint"]:
        path = rd / f"{podcast_id}_{module}.json"
        if path.exists():
            with open(path) as f:
                combined[module] = json.load(f)
        else:
            combined[module] = None

    return combined


def render_comparison_dashboard(
    analyses: List[Dict],
    labels: List[str]
) -> None:
    """
    Render side-by-side comparison of multiple podcast analyses.

    Args:
        analyses : List of combined result dicts
        labels   : Display names for each podcast
    """
    if len(analyses) < 2:
        st.info("Select at least 2 podcasts to compare.")
        return

    st.subheader("📊 Cross-Podcast Comparison")
    st.caption(f"Comparing {len(analyses)} episodes")

    # ── Row 1: KPI Cards ──────────────────────────────
    st.markdown("#### Key Metrics")
    cols = st.columns(len(analyses))

    for col, analysis, label in zip(cols, analyses, labels):
        with col:
            sent_module  = analysis.get("sentiment") or {}
            tone_module  = analysis.get("tone") or {}
            bias_module  = analysis.get("bias") or {}
            ep_module    = analysis.get("emotionprint") or {}

            st.markdown(f"**{label[:30]}**")
            st.metric("Sentiment",    sent_module.get("overall_sentiment","N/A").capitalize())
            st.metric("Score",        f"{sent_module.get('overall_score',0):+.2f}")
            st.metric("Tone",         tone_module.get("dominant_tone","N/A").capitalize())
            st.metric("Bias Score",   f"{bias_module.get('overall_bias_score',0):.0f}/100")
            st.metric("Authenticity", f"{ep_module.get('authenticity_score',100):.0f}%")

    st.divider()

    # ── Row 2: Sentiment Score Bar ────────────────────
    st.markdown("#### Sentiment Score Comparison")
    sent_scores = [
        (a.get("sentiment") or {}).get("overall_score", 0)
        for a in analyses
    ]
    sent_colors = [
        "#27ae60" if s > 0.1 else "#e74c3c" if s < -0.1 else "#95a5a6"
        for s in sent_scores
    ]

    fig_sent = go.Figure(data=[
        go.Bar(
            x=labels,
            y=sent_scores,
            marker_color=sent_colors,
            text=[f"{s:+.2f}" for s in sent_scores],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>"
        )
    ])
    fig_sent.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_sent.update_layout(
        yaxis=dict(range=[-1.1, 1.1], title="Sentiment Score"),
        template="plotly_white",
        height=320,
        showlegend=False
    )
    st.plotly_chart(fig_sent, use_container_width=True)

    # ── Row 3: Bias Score + Tone Comparison ──────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Bias Score Comparison")
        bias_scores = [
            (a.get("bias") or {}).get("overall_bias_score", 0)
            for a in analyses
        ]
        bias_colors = [
            "#e74c3c" if s >= 50 else "#e67e22" if s >= 20 else "#27ae60"
            for s in bias_scores
        ]
        fig_bias = go.Figure(data=[
            go.Bar(
                x=labels,
                y=bias_scores,
                marker_color=bias_colors,
                text=[f"{s:.0f}" for s in bias_scores],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Bias: %{y:.1f}/100<extra></extra>"
            )
        ])
        fig_bias.update_layout(
            yaxis=dict(range=[0, 110], title="Bias Score (0-100)"),
            template="plotly_white",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_bias, use_container_width=True)

    with col_b:
        st.markdown("#### Authenticity Comparison")
        auth_scores = [
            (a.get("emotionprint") or {}).get("authenticity_score", 100.0)
            for a in analyses
        ]
        auth_colors = [
            "#27ae60" if s >= 70 else "#e67e22" if s >= 40 else "#e74c3c"
            for s in auth_scores
        ]
        fig_auth = go.Figure(data=[
            go.Bar(
                x=labels,
                y=auth_scores,
                marker_color=auth_colors,
                text=[f"{s:.0f}%" for s in auth_scores],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Auth: %{y:.0f}%<extra></extra>"
            )
        ])
        fig_auth.update_layout(
            yaxis=dict(range=[0, 115], title="Authenticity (%)"),
            template="plotly_white",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig_auth, use_container_width=True)

    # ── Row 4: Tone Distribution Grouped ─────────────
    st.divider()
    st.markdown("#### Tone Distribution Across Episodes")

    all_tones = ["calm","aggressive","persuasive","anxious","confident","excited"]
    tone_palette = {
        "calm":"#3498db","aggressive":"#e74c3c","persuasive":"#f39c12",
        "anxious":"#9b59b6","confident":"#27ae60","excited":"#1abc9c"
    }

    fig_tone = go.Figure()
    for tone_name in all_tones:
        values = []
        for a in analyses:
            dist = (a.get("tone") or {}).get("tone_distribution",{})
            values.append(dist.get(tone_name,0) * 100)

        if any(v > 0 for v in values):
            fig_tone.add_trace(go.Bar(
                name=tone_name.capitalize(),
                x=labels,
                y=values,
                marker_color=tone_palette[tone_name],
                hovertemplate=f"<b>{tone_name.capitalize()}</b>: %{{y:.1f}}%<extra></extra>"
            ))

    fig_tone.update_layout(
        barmode="group",
        yaxis_title="Percentage (%)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_tone, use_container_width=True)

    # ── Row 5: Sentiment Ratio Side-by-Side ───────────
    st.divider()
    st.markdown("#### Positive / Neutral / Negative Breakdown")

    ratio_data = []
    for a, label in zip(analyses, labels):
        sent = a.get("sentiment") or {}
        ratio_data.append({
            "label":    label,
            "positive": sent.get("positive_ratio",0) * 100,
            "neutral":  sent.get("neutral_ratio",0)  * 100,
            "negative": sent.get("negative_ratio",0) * 100,
        })

    fig_ratio = go.Figure()
    for col_name, color in [("positive","#27ae60"),
                              ("neutral","#95a5a6"),
                              ("negative","#e74c3c")]:
        fig_ratio.add_trace(go.Bar(
            name=col_name.capitalize(),
            x=[r["label"] for r in ratio_data],
            y=[r[col_name] for r in ratio_data],
            marker_color=color,
            hovertemplate=f"<b>{col_name.capitalize()}</b>: %{{y:.1f}}%<extra></extra>"
        ))

    fig_ratio.update_layout(
        barmode="stack",
        yaxis=dict(title="Percentage (%)", range=[0,105]),
        template="plotly_white",
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_ratio, use_container_width=True)
