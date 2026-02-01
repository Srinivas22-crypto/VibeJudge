# create a script: download_models.py
import whisper
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Downloading Whisper model...")
# This will download whisper-small (~244MB)
model = whisper.load_model("small")
print("✓ Whisper-small downloaded")

print("\nDownloading RoBERTa sentiment model...")
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
print("✓ RoBERTa sentiment model downloaded")

print("\nAll models downloaded successfully!")
print(f"Total disk space used: ~750MB")