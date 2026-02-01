import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"
RESULTS_DIR = BASE_DIR / "data" / "results"

# Ensure directories exist
for directory in [UPLOAD_DIR, TRANSCRIPT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File upload settings
MAX_FILE_SIZE_MB = 100
MAX_DURATION_SECONDS = 3600  # 60 minutes
ALLOWED_AUDIO_FORMATS = ['mp3', 'wav', 'm4a', 'ogg', 'flac']

# Model settings
WHISPER_MODEL_SIZE = "small"  # tiny/base/small/medium/large

# Database settings
DATABASE_PATH = str(BASE_DIR / "vibejudge.db")

# UI settings
APP_TITLE = "VibeJudge - Podcast Analyzer"
APP_ICON = "🎙️"
PAGE_LAYOUT = "wide"