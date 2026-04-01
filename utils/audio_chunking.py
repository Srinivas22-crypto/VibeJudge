import os
from pathlib import Path
from pydub import AudioSegment


def ensure_directory(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def split_audio_into_chunks(audio_path: str, chunk_length_ms: int = 5 * 60 * 1000,
                            output_dir: str = "data/chunks") -> list:
    """
    Split audio into smaller chunks.
    Default chunk size = 5 minutes.
    Returns list of chunk file paths.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ensure_directory(output_dir)

    audio = AudioSegment.from_wav(audio_path)
    duration_ms = len(audio)

    chunk_paths = []

    for i, start in enumerate(range(0, duration_ms, chunk_length_ms)):
        end = min(start + chunk_length_ms, duration_ms)
        chunk = audio[start:end]

        chunk_path = os.path.join(output_dir, f"chunk_{i:03d}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    return chunk_paths
