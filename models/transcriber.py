import whisper
import json
from pathlib import Path
import ffmpeg
from config.settings import WHISPER_MODEL_SIZE


class Transcriber:
    """
    Wrapper class for Whisper transcription
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self.model = None

    # ---------------------------------------------------
    # Load Whisper model
    # ---------------------------------------------------
    def load_model(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_size)

    # ---------------------------------------------------
    # Preprocess audio (convert to 16kHz mono WAV)
    # ---------------------------------------------------
    def preprocess_audio(self, file_path: str) -> str:

        path = Path(file_path)
        output_path = path.with_suffix(".processed.wav")

        # If already processed, reuse it
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path)

        try:
            stream = ffmpeg.input(str(path))
            stream = ffmpeg.output(
                stream,
                str(output_path),
                ar="16000",   # sample rate
                ac="1"        # mono
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            return str(output_path)

        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e}")
            return str(path)

        except Exception as e:
            print(f"Preprocessing error: {e}")
            return str(path)

    # ---------------------------------------------------
    # Transcribe audio
    # ---------------------------------------------------
    def transcribe(self, audio_path: str, word_timestamps: bool = False) -> dict:

        self.load_model()

        # preprocess first
        processed_audio = self.preprocess_audio(audio_path)

        result = self.model.transcribe(
            processed_audio,
            word_timestamps=word_timestamps
        )

        # Add useful metadata
        result["word_count"] = len(result.get("text", "").split())

        if result.get("segments"):
            result["duration"] = result["segments"][-1]["end"]
        else:
            result["duration"] = 0

        return result

    # ---------------------------------------------------
    # Save transcript JSON
    # ---------------------------------------------------
    def save_transcript(self, result: dict, output_path: str):

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------
# Singleton instance (for batch processing)
# ---------------------------------------------------

_transcriber_instance = None


def get_transcriber(model_size: str = WHISPER_MODEL_SIZE):

    global _transcriber_instance

    if _transcriber_instance is None:
        _transcriber_instance = Transcriber(model_size)

    return _transcriber_instance