import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
BASE_DIR = Path(__file__).parent
FFMPEG_DIR = BASE_DIR / "ffmpeg"

def download_ffmpeg():
    if (FFMPEG_DIR / "bin" / "ffmpeg.exe").exists():
        print("FFmpeg already exists.")
        return

    print("Downloading FFmpeg...")
    zip_path = BASE_DIR / "ffmpeg.zip"
    try:
        urllib.request.urlretrieve(FFMPEG_URL, zip_path)
        print("Download complete.")
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        
        # Move files to simplify path
        extracted_folder = list(BASE_DIR.glob("ffmpeg-master-latest-win64-gpl"))[0]
        if FFMPEG_DIR.exists():
            shutil.rmtree(FFMPEG_DIR)
        extracted_folder.rename(FFMPEG_DIR)
        
        # Cleanup
        os.remove(zip_path)
        print("FFmpeg setup complete.")
        
    except Exception as e:
        print(f"Error setting up FFmpeg: {e}")

if __name__ == "__main__":
    download_ffmpeg()
