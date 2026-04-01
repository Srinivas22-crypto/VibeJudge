import os
import subprocess
from pathlib import Path


def ensure_directory(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def convert_to_wav_16k_mono(input_path: str, output_dir: str = "data/processed_audio") -> str:
    """
    Convert audio to 16kHz mono WAV using FFmpeg.
    If already converted, reuse the file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    ensure_directory(output_dir)

    input_file = Path(input_path)
    output_path = os.path.join(output_dir, f"{input_file.stem}_16k_mono.wav")

    if os.path.exists(output_path):
        return output_path

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",          
        "-ar", "16000",      
        "-vn",               
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode(errors='ignore')}")

    return output_path
