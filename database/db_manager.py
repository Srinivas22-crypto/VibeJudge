import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json, uuid

class DatabaseManager:
    """Manages all database operations for VibeJudge"""
    
    def __init__(self, db_path: str = "vibejudge.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database and tables if they don't exist"""
        # Read schema
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema = f.read()
        
        # Create database
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Return dict-like rows
        
        cursor = self.connection.cursor()
        cursor.executescript(schema)
        self.connection.commit()
        
        print(f"✓ Database initialized at {self.db_path}")
    
    def insert_podcast(
        self,
        podcast_id: str,
        filename: str,
        original_filename: str,
        file_size: int,
        file_path: str,
        duration: Optional[float] = None
    ) -> bool:
        """
        Insert a new podcast record
        
        Args:
            podcast_id: Unique identifier (UUID)
            filename: Stored filename
            original_filename: User's original filename
            file_size: File size in bytes
            file_path: Absolute path to audio file
            duration: Audio duration in seconds (optional)
        
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO podcasts (
                    id, filename, original_filename, file_size, 
                    file_path, duration, upload_date, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                podcast_id,
                filename,
                original_filename,
                file_size,
                file_path,
                duration,
                datetime.now(),
                'uploaded'
            ))
            self.connection.commit()
            print(f"✓ Podcast {podcast_id} inserted into database")
            return True
        
        except Exception as e:
            print(f"✗ Error inserting podcast: {e}")
            return False
    
    def update_podcast_status(
        self,
        podcast_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update podcast processing status
        
        Args:
            podcast_id: Podcast UUID
            status: New status (uploaded/processing/completed/failed)
            error_message: Error details if status=failed
        
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                UPDATE podcasts 
                SET status = ?, error_message = ?
                WHERE id = ?
            """, (status, error_message, podcast_id))
            self.connection.commit()
            return True
        
        except Exception as e:
            print(f"✗ Error updating status: {e}")
            return False
    
    def get_podcast(self, podcast_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve podcast by ID
        
        Args:
            podcast_id: Podcast UUID
        
        Returns:
            Dictionary with podcast data or None
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM podcasts WHERE id = ?", (podcast_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_recent_podcasts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recently uploaded podcasts
        
        Args:
            limit: Maximum number of podcasts to return
        
        Returns:
            List of podcast dictionaries
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM podcasts 
            ORDER BY upload_date DESC 
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def insert_analysis(self, podcast_id: str, summary: dict) -> str:
        analysis_id = str(uuid.uuid4())[:10]
        self.connection.execute("""
            INSERT INTO analyses(
                analysis_id, podcast_id,
                sentiment_label, sentiment_score, sentiment_confidence,
                dominant_tone, tone_confidence,
                bias_score, bias_level, bias_flags_count,
                authenticity_score, sarcasm_count, suppression_count, irony_count
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            analysis_id, podcast_id,
            summary.get("sentiment_label"), summary.get("sentiment_score"), summary.get("sentiment_confidence"),
            summary.get("dominant_tone"), summary.get("tone_confidence"),
            summary.get("bias_score"), summary.get("bias_level"), summary.get("bias_flags_count"),
            summary.get("authenticity_score"), summary.get("sarcasm_count"),
            summary.get("suppression_count"), summary.get("irony_count"),
        ))
        self.connection.commit()
        return analysis_id

    def insert_bias_flags(self, analysis_id: str, flags: list):
        import uuid, json
        for f in flags:
            self.connection.execute("""
            INSERT INTO bias_flags(
                flag_id, analysis_id, keyword, category, severity,
                timestamp, timestamp_formatted, sentence, text_context, entities_json
          ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
          str(uuid.uuid4())[:12], analysis_id,
          f.get("keyword"), f.get("category"), f.get("severity"),
          f.get("timestamp"), f.get("timestamp_formatted"),
          f.get("sentence"), f.get("text_context"),
          json.dumps(f.get("entities", []))
        ))
        self.connection.commit()

    def insert_emotionprint_flags(self, analysis_id: str, ep_flags: list):
        import uuid, json
        for f in ep_flags:
            self.connection.execute("""
          INSERT INTO emotionprint_flags(
            ep_flag_id, analysis_id, segment_id, timestamp,
            emotional_state, divergence_score, confidence,
            text, prosody_json
          ) VALUES (?,?,?,?,?,?,?,?,?)
        """, (
          str(uuid.uuid4())[:12], analysis_id,
          f.get("segment_id"), f.get("timestamp"),
          f.get("emotional_state"), f.get("divergence_score"),
          f.get("confidence"), f.get("text"),
          json.dumps(f.get("prosody_features", {}))
        ))
        self.connection.commit()
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics
        
        Returns:
            Dictionary with statistics
        """
        cursor = self.connection.cursor()
        
        # Total podcasts
        cursor.execute("SELECT COUNT(*) FROM podcasts")
        total_podcasts = cursor.fetchone()[0]
        
        # Total analyses
        cursor.execute("SELECT COUNT(*) FROM analyses")
        total_analyses = cursor.fetchone()[0]
        
        # Average bias score
        cursor.execute("SELECT AVG(bias_score) FROM analyses")
        avg_bias = cursor.fetchone()[0] or 0
        
        # Average sentiment score
        cursor.execute("SELECT AVG(sentiment_score) FROM analyses")
        avg_sentiment = cursor.fetchone()[0] or 0
        
        return {
            'total_podcasts': total_podcasts,
            'total_analyses': total_analyses,
            'avg_bias_score': round(avg_bias, 2),
            'avg_sentiment_score': round(avg_sentiment, 2)
        }
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✓ Database connection closed")


# Singleton instance
_db_instance = None

def get_db() -> DatabaseManager:
    """Get global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance