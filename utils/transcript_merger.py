import json
from pathlib import Path


def merge_chunk_transcripts(chunk_results: list, chunk_duration_sec: int = 300):
    """
    Merge chunk-level transcripts into one unified transcript.
    chunk_duration_sec default = 300 seconds = 5 min
    """
    full_text = []
    merged_segments = []

    for i, chunk_data in enumerate(chunk_results):
        result = chunk_data["result"]
        offset = i * chunk_duration_sec

        full_text.append(result["text"])

        for seg in result["segments"]:
            merged_segments.append({
                "start": seg["start"] + offset,
                "end": seg["end"] + offset,
                "text": seg["text"]
            })

    return {
        "text": " ".join(full_text),
        "segments": merged_segments
    }


def save_merged_transcript(transcript: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
