from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SentimentResult(BaseModel):
    overall_sentiment: str
    confidence: float
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float

class BiasResult(BaseModel):
    overall_bias_score: float
    bias_level: str
    flags_count: int
    category_distribution: Dict[str, int]

class AnalysisResponse(BaseModel):
    podcast_id: str
    filename: str
    duration_seconds: float
    transcription_text: str
    sentiment: Optional[SentimentResult]
    bias: Optional[BiasResult]
    processing_time_seconds: float
    created_at: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    expires_in: int
