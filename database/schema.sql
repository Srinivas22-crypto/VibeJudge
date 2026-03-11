-- Podcasts table: stores uploaded audio metadata
CREATE TABLE IF NOT EXISTS podcasts (
    id TEXT PRIMARY KEY,                    -- UUID
    filename TEXT NOT NULL,                 -- Original filename
    original_filename TEXT,                 -- User's filename
    file_size INTEGER,                      -- Bytes
    duration REAL,                          -- Seconds
    language TEXT DEFAULT 'en',
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    file_path TEXT,                         -- Path to audio file
    transcript_path TEXT,                   -- Path to transcript JSON
    status TEXT DEFAULT 'uploaded',         -- uploaded/processing/completed/failed
    error_message TEXT                      -- If status=failed
);

CREATE TABLE IF NOT EXISTS analyses (
  analysis_id TEXT PRIMARY KEY,
  podcast_id TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

  sentiment_label TEXT,
  sentiment_score REAL,
  sentiment_confidence REAL,

  dominant_tone TEXT,
  tone_confidence REAL,

  bias_score REAL,
  bias_level TEXT,
  bias_flags_count INTEGER,

  authenticity_score REAL,
  sarcasm_count INTEGER,
  suppression_count INTEGER,
  irony_count INTEGER,

  FOREIGN KEY(podcast_id) REFERENCES podcasts(podcast_id)
);

CREATE TABLE IF NOT EXISTS bias_flags (
  flag_id TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL,
  keyword TEXT,
  category TEXT,
  severity TEXT,
  timestamp REAL,
  timestamp_formatted TEXT,
  sentence TEXT,
  text_context TEXT,
  entities_json TEXT,
  FOREIGN KEY(analysis_id) REFERENCES analyses(analysis_id)
);

CREATE TABLE IF NOT EXISTS emotionprint_flags (
  ep_flag_id TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL,
  segment_id INTEGER,
  timestamp TEXT,
  emotional_state TEXT,
  divergence_score REAL,
  confidence REAL,
  text TEXT,
  prosody_json TEXT,
  FOREIGN KEY(analysis_id) REFERENCES analyses(analysis_id)
);

CREATE TABLE IF NOT EXISTS performance_runs (
  perf_id TEXT PRIMARY KEY,
  analysis_id TEXT NOT NULL,
  total_duration_s REAL,
  realtime_factor REAL,
  peak_memory_mb REAL,
  stages_json TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(analysis_id) REFERENCES analyses(analysis_id)
);
