# models/bias_detector_v2.py

import json
import os
import numpy as np
import spacy
from transformers import pipeline


class BiasDetectorV2:
    """
    Hybrid bias detection:
    1. keyword/category heuristic
    2. sentence context scoring
    3. toxicity/bias prior from pretrained model
    """

    def __init__(self, keyword_file="data/bias_keywords.json",
                 classifier_model="s-nlp/roberta_toxicity_classifier"):
        self.nlp = spacy.load("en_core_web_sm")

        if not os.path.exists(keyword_file):
            raise FileNotFoundError(f"Bias keyword file not found: {keyword_file}")

        with open(keyword_file, "r", encoding="utf-8") as f:
            self.bias_keywords = json.load(f)

        self.classifier = pipeline(
            "text-classification",
            model=classifier_model,
            truncation=True
        )

    def split_sentences(self, text):
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def get_entities(self, text):
        doc = self.nlp(text)
        return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

    def heuristic_match(self, sentence):
        sentence_lower = sentence.lower()
        matches = []

        for category, keywords in self.bias_keywords.items():
            for kw in keywords:
                if kw.lower() in sentence_lower:
                    matches.append({
                        "keyword": kw,
                        "category": category
                    })

        return matches

    def heuristic_score(self, matches):
        if not matches:
            return 0.0

        # light weighting by count
        score = min(1.0, 0.2 + 0.2 * len(matches))
        return round(score, 4)

    def model_score(self, sentence):
        """
        Use toxicity classifier as a soft signal.
        Not all toxicity = bias, so we use it carefully.
        """
        result = self.classifier(sentence)[0]
        label = result["label"].lower()
        score = float(result["score"])

        # normalize label interpretations
        if "toxic" in label or label in ["1", "label_1"]:
            return score
        return 1.0 - score if label in ["0", "label_0", "non-toxic", "not toxic"] else score

    def combine_scores(self, heuristic_score, model_score):
        """
        Weighted fusion:
        Heuristic is primary because category explainability matters.
        Model acts as contextual correctness prior.
        """
        final = 0.65 * heuristic_score + 0.35 * model_score
        return round(min(1.0, final), 4)

    def classify_level(self, score):
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0:
            return "LOW"
        return "NONE"

    def align_with_segments(self, flagged_instances, transcript_segments):
        seg_count = len(transcript_segments)
        inst_count = len(flagged_instances)

        if seg_count == 0:
            return flagged_instances

        aligned = []
        for i, item in enumerate(flagged_instances):
            seg_idx = min(int(i * seg_count / max(1, inst_count)), seg_count - 1)
            seg = transcript_segments[seg_idx]
            aligned.append({
                **item,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0)
            })
        return aligned

    def generate_timeline(self, instances, bin_size=30):
        bins = {}
        for item in instances:
            t = int(item.get("start", 0) // bin_size) * bin_size
            bins[t] = bins.get(t, 0) + 1

        return [
            {
                "time_sec": t,
                "time_label": f"{t//60:02d}:{t%60:02d}",
                "count": count
            }
            for t, count in sorted(bins.items())
        ]

    def analyze(self, transcript_text, transcript_segments):
        sentences = self.split_sentences(transcript_text)
        flagged = []

        for sent in sentences:
            matches = self.heuristic_match(sent)
            if not matches:
                continue

            h_score = self.heuristic_score(matches)
            m_score = self.model_score(sent)
            final_score = self.combine_scores(h_score, m_score)
            level = self.classify_level(final_score)

            if final_score >= 0.2:
                flagged.append({
                    "sentence": sent,
                    "matches": matches,
                    "heuristic_score": h_score,
                    "model_score": round(m_score, 4),
                    "final_score": final_score,
                    "level": level,
                    "entities": self.get_entities(sent)
                })

        aligned = self.align_with_segments(flagged, transcript_segments)
        timeline = self.generate_timeline(aligned)

        total_flags = len(aligned)
        overall_score = round(
            min(100.0, np.mean([x["final_score"] for x in aligned]) * 100) if aligned else 0.0,
            2
        )

        category_counts = {}
        for item in aligned:
            for m in item["matches"]:
                category_counts[m["category"]] = category_counts.get(m["category"], 0) + 1

        total_cats = sum(category_counts.values()) or 1
        category_distribution = {
            k: round(v * 100 / total_cats, 2) for k, v in category_counts.items()
        }

        overall_level = self.classify_level(overall_score / 100.0)

        return {
            "summary": {
                "bias_score": overall_score,
                "bias_level": overall_level,
                "total_flags": total_flags
            },
            "category_distribution": category_distribution,
            "timeline": timeline,
            "flagged_instances": aligned
        }
