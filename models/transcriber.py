import whisper
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Suppress whisper warnings
warnings.filterwarnings("ignore")

class AudioTranscriber:
    """Handles audio transcription using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "small"):
        """
        Initialize Whisper model
        """
        print(f"Loading Whisper model ({model_size})...")
        self.model = whisper.load_model(model_size)
        print(f"✓ Whisper model loaded")
        
        # Verify ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: ffmpeg not found in PATH. Audio processing may fail.")
            # Try to add local ffmpeg to path if it exists
            local_ffmpeg = Path(__file__).parent.parent / "ffmpeg" / "bin"
            if local_ffmpeg.exists():
                os.environ["PATH"] += os.pathsep + str(local_ffmpeg)
                print(f"Added local ffmpeg to PATH: {local_ffmpeg}")
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio duration in seconds using ffprobe
        """
        try:
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                audio_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            print(f"Warning: Could not get duration: {e}")
            return 0.0
    
    def preprocess_audio(
        self,
        input_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Preprocess audio file using ffmpeg directly
        - Convert to mono
        - Resample to 16kHz
        """
        print(f"Preprocessing audio: {input_path}")
        
        try:
            if output_path is None:
                base_path = Path(input_path)
                output_path = str(base_path.parent / f"{base_path.stem}_processed.wav")
            
            # FFmpeg command: -y (overwrite), -i input, -ac 1 (mono), -ar 16000 (sample rate)
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-ac", "1",
                "-ar", "16000",
                output_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            print(f"✓ Audio preprocessed: {output_path}")
            return output_path
        
        except subprocess.CalledProcessError as e:
            print(f"✗ Preprocessing failed: {e}")
            print(f"  Using original file: {input_path}")
            return input_path
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        task: str = "transcribe",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file
        """
        print(f"\nTranscribing: {audio_path}")
        
        try:
            # Get duration
            duration = self.get_audio_duration(audio_path)
            print(f"Audio duration: {duration:.1f} seconds")
            
            # Transcribe
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                verbose=verbose,
                word_timestamps=True
            )
            
            # Add duration to result
            result['duration'] = duration
            
            print(f"✓ Transcription complete")
            return result
        
        except Exception as e:
            print(f"✗ Transcription failed: {e}")
            raise
    
    def save_transcript(self, transcript: Dict[str, Any], output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            print(f"✓ Transcript saved: {output_path}")
            return True
        except Exception as e:
            print(f"✗ Failed to save transcript: {e}")
            return False
    
    def load_transcript(self, json_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"✗ Failed to load transcript: {e}")
            return None
    
    def format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def get_transcript_summary(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        text = transcript.get('text', '')
        segments = transcript.get('segments', [])
        return {
            'total_words': len(text.split()),
            'total_characters': len(text),
            'total_segments': len(segments),
            'duration': transcript.get('duration', 0),
            'language': transcript.get('language', 'unknown'),
            'avg_confidence': sum(seg.get('avg_logprob', 0) for seg in segments) / len(segments) if segments else 0
        }

# Singleton instance
_transcriber_instance = None

def get_transcriber() -> AudioTranscriber:
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = AudioTranscriber()
    return _transcriber_instance