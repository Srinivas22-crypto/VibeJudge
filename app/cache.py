import streamlit as st

@st.cache_resource
def get_whisper_model(size: str):
    import whisper
    return whisper.load_model(size)

@st.cache_resource
def get_sentiment_pipeline():
    from transformers import pipeline
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=-1
    )

@st.cache_resource
def get_spacy_model():
    from utils.spacy_utils import load_spacy_model
    return load_spacy_model("en_core_web_sm")
