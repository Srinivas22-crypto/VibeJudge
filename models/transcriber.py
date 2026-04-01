from faster_whisper import WhisperModel
import os
import json


class FastTranscriber:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_chunk(self, audio_path: str):
        """
        Transcribe a single chunk.
        """
        segments, info = self.model.transcribe(audio_path, beam_size=1)

        segment_list = []
        full_text = []

        for segment in segments:
            segment_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            }
            segment_list.append(segment_data)
            full_text.append(segment.text.strip())

        return {
            "text": " ".join(full_text),
            "segments": segment_list,
            "language": info.language if info else "unknown"
        }

    def save_transcript(self, result: dict, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
