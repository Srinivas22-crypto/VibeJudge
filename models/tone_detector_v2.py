# models/tone_detector_v2.py

import numpy as np
import spacy
from transformers import pipeline


class ToneDetectorV2:
    """
    Multi-label tone detector using zero-shot classification.
    """

    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name
        )
        self.labels = [
            "informative",
            "neutral",
            "emotional",
            "aggressive",
            "persuasive",
            "confident",
            "anxious",
            "excited",
            "calm"
        ]

    def split_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def predict_sentence(self, sentence):
        result = self.classifier(
            sentence,
            candidate_labels=self.labels,
            multi_label=True
        )

        pairs = list(zip(result["labels"], result["scores"]))
        return {label: round(float(score), 4) for label, score in pairs}

    def align_with_segments(self, sentence_results, transcript_segments):
        seg_count = len(transcript_segments)
        sent_count = len(sentence_results)

        if seg_count == 0:
            return sentence_results

        aligned = []
        for i, item in enumerate(sentence_results):
            seg_idx = min(int(i * seg_count / sent_count), seg_count - 1)
            seg = transcript_segments[seg_idx]

            aligned.append({
                **item,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0)
            })

        return aligned

    def summarize(self, sentence_results):
        if not sentence_results:
            return {
                "dominant_tone": "neutral",
                "tone_score": 0.0,
                "confidence": 0.0,
                "distribution": {}
            }

        aggregate = {label: [] for label in self.labels}

        for item in sentence_results:
            for label, score in item["tones"].items():
                aggregate[label].append(score)

        avg_scores = {label: round(float(np.mean(scores)), 4) for label, scores in aggregate.items()}
        dominant_tone = max(avg_scores, key=avg_scores.get)
        confidence = avg_scores[dominant_tone]

        distribution = {
            label: round(score * 100, 2) for label, score in avg_scores.items()
        }

        examples = {}
        for label in self.labels:
            label_examples = sorted(
                sentence_results,
                key=lambda x: x["tones"].get(label, 0),
                reverse=True
            )[:2]
            examples[label] = label_examples

        return {
            "dominant_tone": dominant_tone,
            "tone_score": round(avg_scores[dominant_tone], 4),
            "confidence": round(confidence, 4),
            "distribution": distribution,
            "examples": examples
        }

    def analyze(self, transcript_text, transcript_segments):
        sentences = self.split_sentences(transcript_text)

        sentence_results = []
        for sent in sentences:
            tones = self.predict_sentence(sent)
            sentence_results.append({
                "text": sent,
                "tones": tones
            })

        aligned = self.align_with_segments(sentence_results, transcript_segments)
        summary = self.summarize(aligned)

        return {
            "summary": summary,
            "sentence_results": aligned
        }
