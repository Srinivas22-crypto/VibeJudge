import sqlite3
import os
from pathlib import Path

# Config
BASE_DIR = Path(__file__).parent.parent
DB_PATH = BASE_DIR / "vibejudge.db"
SCHEMA_PATH = BASE_DIR / "database" / "schema.sql"

def init_db():
    print(f"Initializing database at: {DB_PATH}")
    
    # Read schema
    if not SCHEMA_PATH.exists():
        print(f"Error: Schema file not found at {SCHEMA_PATH}")
        return

    with open(SCHEMA_PATH, 'r') as f:
        schema_sql = f.read()

    try:
        # Connect and execute
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Execute script
        cursor.executescript(schema_sql)
        print("Schema executed successfully.")
        
        # Verify tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Created tables:", [t[0] for t in tables])
        
        conn.commit()
        conn.close()
        print("Database initialization complete.")
        
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")

if __name__ == "__main__":
    init_db()
