import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

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
    
    def insert_analysis(
        self,
        podcast_id: str,
        sentiment_data: Dict[str, Any],
        tone_data: Dict[str, Any],
        bias_data: Dict[str, Any],
        processing_time: float,
        result_json_path: str
    ) -> Optional[int]:
        """
        Insert analysis results
        
        Args:
            podcast_id: Associated podcast UUID
            sentiment_data: Dictionary with sentiment metrics
            tone_data: Dictionary with tone metrics
            bias_data: Dictionary with bias metrics
            processing_time: Total processing time in seconds
            result_json_path: Path to full results JSON
        
        Returns:
            Analysis ID if successful, None otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO analyses (
                    podcast_id,
                    sentiment_positive_pct, sentiment_neutral_pct, sentiment_negative_pct,
                    sentiment_score,
                    dominant_tone,
                    tone_calm_pct, tone_aggressive_pct, tone_persuasive_pct,
                    tone_anxious_pct, tone_confident_pct, tone_excited_pct,
                    bias_score, bias_level, bias_flags_count,
                    processing_time, result_json_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                podcast_id,
                sentiment_data.get('positive_pct', 0),
                sentiment_data.get('neutral_pct', 0),
                sentiment_data.get('negative_pct', 0),
                sentiment_data.get('overall_score', 0),
                tone_data.get('dominant_tone', 'Unknown'),
                tone_data.get('calm_pct', 0),
                tone_data.get('aggressive_pct', 0),
                tone_data.get('persuasive_pct', 0),
                tone_data.get('anxious_pct', 0),
                tone_data.get('confident_pct', 0),
                tone_data.get('excited_pct', 0),
                bias_data.get('score', 0),
                bias_data.get('level', 'Unknown'),
                bias_data.get('flags_count', 0),
                processing_time,
                result_json_path
            ))
            
            analysis_id = cursor.lastrowid
            self.connection.commit()
            
            print(f"✓ Analysis {analysis_id} inserted for podcast {podcast_id}")
            return analysis_id
        
        except Exception as e:
            print(f"✗ Error inserting analysis: {e}")
            return None
    
    def insert_bias_flags(
        self,
        analysis_id: int,
        flags: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert bias flags for an analysis
        
        Args:
            analysis_id: Associated analysis ID
            flags: List of bias flag dictionaries
        
        Returns:
            True if successful
        """
        try:
            cursor = self.connection.cursor()
            
            for flag in flags:
                cursor.execute("""
                    INSERT INTO bias_flags (
                        analysis_id, phrase, category, severity,
                        sentence, context, timestamp, timestamp_seconds
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    flag.get('phrase', ''),
                    flag.get('category', ''),
                    flag.get('severity', 'medium'),
                    flag.get('sentence', ''),
                    flag.get('context', ''),
                    flag.get('timestamp', '00:00'),
                    flag.get('timestamp_seconds', 0.0)
                ))
            
            self.connection.commit()
            print(f"✓ {len(flags)} bias flags inserted for analysis {analysis_id}")
            return True
        
        except Exception as e:
            print(f"✗ Error inserting bias flags: {e}")
            return False
    
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