import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
from config.settings import Config
from database.db_manager import DatabaseManager
from api.schemas.upload_schema import UploadResponse

router = APIRouter(prefix="/upload", tags=["Upload"])
config = Config()
db = DatabaseManager()

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac"}
MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

@router.post("/", response_model=UploadResponse)
async def upload_podcast(file: UploadFile = File(...)):
    """
    Upload a podcast audio file for analysis.
    Supports MP3, WAV, M4A, OGG, FLAC up to 100 MB.
    """
    # Validate extension
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    # Read and check size
    content = await file.read()
    file_size = len(content)
    if file_size > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413,
            detail=f"File too large ({file_size/1e6:.1f} MB). Max: 100 MB")

    # Save file
    podcast_id = str(uuid.uuid4())
    save_path = os.path.join(config.UPLOAD_DIR, f"{podcast_id}.{ext}")
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(content)

    # Save to DB
    db.insert_podcast({
        "id": podcast_id,
        "filename": file.filename,
        "file_path": save_path,
        "file_size": file_size,
        "status": "pending",
        "upload_date": datetime.now().isoformat()
    })

    return UploadResponse(
        success=True,
        podcast_id=podcast_id,
        filename=file.filename,
        file_size_mb=round(file_size / 1e6, 2),
        message="File uploaded successfully. Submit to /analysis/start to begin."
    )

@router.get("/status/{podcast_id}")
async def get_upload_status(podcast_id: str):
    """Check upload and processing status."""
    record = db.get_podcast(podcast_id)
    if not record:
        raise HTTPException(status_code=404, detail="Podcast not found")
    return {"podcast_id": podcast_id, "status": record.get("status"), 
            "filename": record.get("filename")}
