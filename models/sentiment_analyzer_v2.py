# models/sentiment_analyzer_v2.py

import numpy as np
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax


class SentimentAnalyzerV2:
    """
    Transformer-based sentiment analysis using RoBERTa.
    Produces:
    - sentence-level labels
    - confidence
    - overall score
    - timeline bins
    - most positive / negative moments
    """

    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.label_map = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE"
        }
        self.score_map = {
            "NEGATIVE": -1.0,
            "NEUTRAL": 0.0,
            "POSITIVE": 1.0
        }
        self.nlp = spacy.load("en_core_web_sm")

    def split_sentences(self, text: str):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def predict_batch(self, texts, batch_size=8):
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

            with torch.no_grad():
                output = self.model(**encoded)
                probs = softmax(output.logits.detach().cpu().numpy(), axis=1)

            for text, prob in zip(batch, probs):
                label_idx = int(np.argmax(prob))
                label = self.label_map[label_idx]
                confidence = float(np.max(prob))
                score = float(
                    prob[2] * 1.0 +   # positive
                    prob[1] * 0.0 +   # neutral
                    prob[0] * -1.0    # negative
                )

                all_results.append({
                    "text": text,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "score": round(score, 4),
                    "probs": {
                        "negative": round(float(prob[0]), 4),
                        "neutral": round(float(prob[1]), 4),
                        "positive": round(float(prob[2]), 4)
                    }
                })

        return all_results

    def align_sentences_with_segments(self, sentence_results, transcript_segments):
        """
        Simple approximate alignment:
        assign each sentence to the nearest segment based on order.
        """
        aligned = []
        seg_count = len(transcript_segments)
        sent_count = len(sentence_results)

        if seg_count == 0:
            return sentence_results

        for i, result in enumerate(sentence_results):
            seg_idx = min(int(i * seg_count / sent_count), seg_count - 1)
            segment = transcript_segments[seg_idx]

            aligned.append({
                **result,
                "start": segment.get("start", 0),
                "end": segment.get("end", 0)
            })

        return aligned

    def generate_timeline(self, aligned_results, bin_size=30):
        bins = {}

        for item in aligned_results:
            start = item.get("start", 0)
            bin_key = int(start // bin_size) * bin_size

            if bin_key not in bins:
                bins[bin_key] = []

            bins[bin_key].append(item["score"])

        timeline = []
        for t in sorted(bins.keys()):
            avg_score = float(np.mean(bins[t]))
            timeline.append({
                "time_sec": t,
                "time_label": f"{t//60:02d}:{t%60:02d}",
                "score": round(avg_score, 4)
            })

        return timeline

    def summarize(self, aligned_results):
        if not aligned_results:
            return {
                "overall_label": "NEUTRAL",
                "overall_score": 0.0,
                "confidence": 0.0,
                "sentence_count": 0
            }

        scores = [x["score"] for x in aligned_results]
        confidences = [x["confidence"] for x in aligned_results]

        overall_score = float(np.mean(scores))
        avg_conf = float(np.mean(confidences))

        if overall_score > 0.1:
            overall_label = "POSITIVE"
        elif overall_score < -0.1:
            overall_label = "NEGATIVE"
        else:
            overall_label = "NEUTRAL"

        label_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        for item in aligned_results:
            label_counts[item["label"]] += 1

        total = len(aligned_results)

        sorted_positive = sorted(aligned_results, key=lambda x: x["score"], reverse=True)
        sorted_negative = sorted(aligned_results, key=lambda x: x["score"])

        return {
            "overall_label": overall_label,
            "overall_score": round(overall_score, 4),
            "confidence": round(avg_conf, 4),
            "sentence_count": total,
            "distribution": {
                "positive_pct": round(100 * label_counts["POSITIVE"] / total, 2),
                "neutral_pct": round(100 * label_counts["NEUTRAL"] / total, 2),
                "negative_pct": round(100 * label_counts["NEGATIVE"] / total, 2)
            },
            "most_positive": sorted_positive[:3],
            "most_negative": sorted_negative[:3]
        }

    def analyze(self, transcript_text: str, transcript_segments: list):
        sentences = self.split_sentences(transcript_text)
        sentence_results = self.predict_batch(sentences)
        aligned = self.align_sentences_with_segments(sentence_results, transcript_segments)
        timeline = self.generate_timeline(aligned)
        summary = self.summarize(aligned)

        return {
            "summary": summary,
            "sentence_results": aligned,
            "timeline": timeline
        }
