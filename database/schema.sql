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

-- Analyses table: stores complete analysis results
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    podcast_id TEXT NOT NULL,
    
    -- Sentiment metrics
    sentiment_positive_pct REAL,            -- Percentage 0-100
    sentiment_neutral_pct REAL,
    sentiment_negative_pct REAL,
    sentiment_score REAL,                   -- Overall score -1 to +1
    
    -- Tone metrics
    dominant_tone TEXT,                     -- calm/aggressive/persuasive/etc
    tone_calm_pct REAL,
    tone_aggressive_pct REAL,
    tone_persuasive_pct REAL,
    tone_anxious_pct REAL,
    tone_confident_pct REAL,
    tone_excited_pct REAL,
    
    -- Bias metrics
    bias_score INTEGER,                     -- 0-100
    bias_level TEXT,                        -- Low/Moderate/High
    bias_flags_count INTEGER,
    
    -- Processing metadata
    processing_time REAL,                   -- Seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    result_json_path TEXT,                  -- Path to full results JSON
    
    FOREIGN KEY (podcast_id) REFERENCES podcasts(id) ON DELETE CASCADE
);

-- Bias flags table: stores individual biased phrases
CREATE TABLE IF NOT EXISTS bias_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    phrase TEXT NOT NULL,
    category TEXT NOT NULL,                 -- political_left/political_right/gender/loaded/weasel
    severity TEXT,                          -- low/medium/high
    sentence TEXT,                          -- Full sentence containing phrase
    context TEXT,                           -- 2 sentences before + after
    timestamp TEXT,                         -- Format: "MM:SS"
    timestamp_seconds REAL,                 -- For sorting/filtering
    
    FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
);

-- Index for faster queries
CREATE INDEX IF NOT EXISTS idx_podcasts_status ON podcasts(status);
CREATE INDEX IF NOT EXISTS idx_podcasts_upload_date ON podcasts(upload_date DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_podcast_id ON analyses(podcast_id);
CREATE INDEX IF NOT EXISTS idx_bias_flags_analysis_id ON bias_flags(analysis_id);