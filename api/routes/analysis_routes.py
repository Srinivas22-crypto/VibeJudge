import os
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks
from database.db_manager import DatabaseManager
from models.transcriber import Transcriber
from models.sentiment_analyzer import SentimentAnalyzer
from models.bias_detector import BiasDetector

router = APIRouter(prefix="/analysis", tags=["Analysis"])
db = DatabaseManager()

def run_full_analysis(podcast_id: str, file_path: str):
    """Background task: run full pipeline."""
    try:
        db.update_podcast_status(podcast_id, "processing")

        # Stage 1: Transcription
        transcriber = Transcriber()
        result = transcriber.transcribe(file_path, word_timestamps=True)
        transcript_text = result["text"]
        segments = result.get("segments", [])

        # Stage 2: Sentiment
        analyzer = SentimentAnalyzer()
        sentiment = analyzer.analyze_text(transcript_text)

        # Stage 3: Bias Detection
        detector = BiasDetector()
        bias = detector.analyze_text(transcript_text, segments=segments)

        # Save combined result
        from config.settings import Config
        config = Config()
        out_path = os.path.join(config.RESULTS_DIR, f"{podcast_id}_full_analysis.json")
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "podcast_id": podcast_id,
                "transcript": transcript_text,
                "segments": segments,
                "sentiment": sentiment,
                "bias": bias
            }, f, indent=2, default=str)

        db.update_podcast_status(podcast_id, "completed", result_path=out_path)

    except Exception as e:
        db.update_podcast_status(podcast_id, "failed", error=str(e))

@router.post("/start/{podcast_id}")
async def start_analysis(podcast_id: str, background_tasks: BackgroundTasks):
    """Trigger full pipeline analysis for an uploaded podcast."""
    record = db.get_podcast(podcast_id)
    if not record:
        raise HTTPException(status_code=404, detail="Podcast not found")
    if record["status"] not in ("pending", "failed"):
        return {"message": f"Analysis already {record['status']}"}

    file_path = record["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file missing from disk")

    background_tasks.add_task(run_full_analysis, podcast_id, file_path)
    return {"message": "Analysis started", "podcast_id": podcast_id}

@router.get("/result/{podcast_id}")
async def get_analysis_result(podcast_id: str):
    """Retrieve the completed analysis result."""
    record = db.get_podcast(podcast_id)
    if not record:
        raise HTTPException(status_code=404, detail="Podcast not found")

    if record["status"] != "completed":
        return {"podcast_id": podcast_id, "status": record["status"],
                "message": "Analysis not yet complete"}

    result_path = record.get("result_path")
    if not result_path or not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Result file not found")

    with open(result_path, "r") as f:
        data = json.load(f)
    return data

@router.get("/list")
async def list_analyses():
    """List all podcasts and their analysis status."""
    return db.list_podcasts()
