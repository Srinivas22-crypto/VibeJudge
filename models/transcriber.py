import whisper
import os
import json
from pathlib import Path
import ffmpeg
import streamlit as st
from config.settings import WHISPER_MODEL_SIZE

class Transcriber:
    def __init__(self):
        self.model_size = WHISPER_MODEL_SIZE
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_size)
    
    def preprocess_audio(self, file_path: str) -> str:
        """
        Convert audio to 16kHz WAV which is optimal for Whisper
        """
        path = Path(file_path)
        output_path = path.with_suffix('.processed.wav')
        
        # If already exists and not empty, return it
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)
            
        try:
            # Convert to 16kHz mono wav
            stream = ffmpeg.input(str(path))
            stream = ffmpeg.output(stream, str(output_path), ar='16000', ac='1')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            return str(output_path)
        except ffmpeg.Error as e:
            # If ffmpeg fails, return original path and hope for the best
            # or raise an error if it's critical
            print(f"FFmpeg error: {e}")
            return str(path)
        except Exception as e:
            print(f"Error preprocessing: {e}")
            return str(path)

    def transcribe(self, audio_path: str) -> dict:
        self.load_model()
        result = self.model.transcribe(audio_path)
        return result

    def save_transcript(self, result: dict, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

# Singleton instance
_transcriber_instance = None

def get_transcriber():
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = Transcriber()
    return _transcriber_instance
