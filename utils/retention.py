from pathlib import Path
from datetime import datetime, timedelta
import os

def delete_older_than(folder: str, days: int = 7):
    base = Path(folder)
    if not base.exists():
        return 0

    cutoff = datetime.now() - timedelta(days=days)
    deleted = 0
    for p in base.glob("*"):
        if not p.is_file():
            continue
        mtime = datetime.fromtimestamp(p.stat().st_mtime)
        if mtime < cutoff:
            try:
                p.unlink()
                deleted += 1
            except:
                pass
    return deleted
