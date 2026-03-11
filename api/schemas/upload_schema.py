from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    success: bool
    podcast_id: str
    filename: str
    file_size_mb: float
    message: str

class AnalysisStatusResponse(BaseModel):
    podcast_id: str
    status: str           # pending | processing | completed | failed
    progress: int         # 0-100
    current_stage: Optional[str]
    error: Optional[str]
