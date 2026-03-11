import io
import logging
from collections import Counter
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def generate_transcript_wordcloud(
    transcript_text: str,
    max_words: int = 100
) -> Optional[io.BytesIO]:
    """
    Generate word cloud from transcript text.

    Args:
        transcript_text : Full transcript string
        max_words       : Maximum number of words to display

    Returns:
        BytesIO PNG buffer or None if wordcloud not installed
    """
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        logger.warning("wordcloud not installed. Run: pip install wordcloud")
        return None

    # Extended stopwords for podcast context
    stopwords = set(STOPWORDS) | {
        "um","uh","like","you","know","going","gonna","wanna",
        "thing","things","really","just","right","okay","well",
        "actually","basically","literally","also","even","still",
        "think","thought","say","said","told","know","want"
    }

    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        stopwords=stopwords,
        max_words=max_words,
        colormap="Blues",
        collocations=True,
        prefer_horizontal=0.85,
        margin=10
    )

    try:
        wc.generate(transcript_text)
    except ValueError:
        logger.warning("Text too short for word cloud")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Transcript Word Cloud", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_bias_wordcloud(bias_flags: List[Dict]) -> Optional[io.BytesIO]:
    """
    Generate word cloud weighted by bias flag frequency.

    Args:
        bias_flags: List of bias flag dicts from BiasDetector

    Returns:
        BytesIO PNG buffer or None
    """
    if not bias_flags:
        return None

    try:
        from wordcloud import WordCloud
    except ImportError:
        return None

    # Build frequency dict from bias keywords
    keyword_counts = Counter()
    for flag in bias_flags:
        kw = flag.get("keyword","")
        if kw:
            keyword_counts[kw] += 1

    if not keyword_counts:
        return None

    # Category-based color mapping
    category_colors = {
        "political_left":   "#3498db",
        "political_right":  "#e74c3c",
        "gender_bias":      "#9b59b6",
        "loaded_language":  "#e67e22",
        "weasel_words":     "#95a5a6"
    }

    # Map keyword → hex color
    keyword_colors = {}
    for flag in bias_flags:
        kw  = flag.get("keyword","")
        cat = flag.get("category","")
        if kw:
            keyword_colors[kw] = category_colors.get(cat,"#2c3e50")

    def color_func(word, **kwargs):
        hex_c = keyword_colors.get(word.lower(), "#2c3e50")
        r = int(hex_c[1:3], 16)
        g = int(hex_c[3:5], 16)
        b = int(hex_c[5:7], 16)
        return f"rgb({r},{g},{b})"

    wc = WordCloud(
        width=800, height=350,
        background_color="white",
        color_func=color_func,
        prefer_horizontal=0.8
    )
    wc.generate_from_frequencies(dict(keyword_counts))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Bias Keywords Word Cloud", fontsize=13, fontweight="bold", pad=10)

    # Legend
    patches = [
        mpatches.Patch(color=c, label=k.replace("_"," ").title())
        for k, c in category_colors.items()
    ]
    ax.legend(handles=patches, loc="lower right",
              fontsize=7, ncol=2, framealpha=0.8)
    import matplotlib.patches as mpatches

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_frequency_chart(
    transcript_text: str,
    top_n: int = 20
) -> io.BytesIO:
    """
    Horizontal bar chart of top N most frequent meaningful words.

    Args:
        transcript_text : Full transcript text
        top_n           : Number of top words to show

    Returns:
        BytesIO PNG buffer
    """
    stopwords = {
        "the","a","an","and","or","but","in","on","at","to","for",
        "of","with","by","from","up","about","into","through","is",
        "are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might",
        "this","that","these","those","it","its","we","they","you","he","she",
        "i","me","my","our","your","their","his","her","um","uh","like"
    }

    words = [
        w.lower().strip(".,!?\"'()[]") 
        for w in transcript_text.split()
        if len(w) > 3
    ]
    words = [w for w in words if w not in stopwords and w.isalpha()]
    counter = Counter(words).most_common(top_n)

    if not counter:
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)
        return buf

    labels = [c[0] for c in reversed(counter)]
    values = [c[1] for c in reversed(counter)]

    cmap = plt.cm.Blues
    bar_colors = [cmap(0.4 + 0.6 * (i / len(values))) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.35)))
    bars = ax.barh(labels, values, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=7.5)

    ax.set_xlabel("Frequency", fontsize=9)
    ax.set_title(f"Top {top_n} Most Frequent Words", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf
