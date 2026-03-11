import sqlite3
import uuid
from datetime import datetime
from auth.password_utils import hash_password, verify_password

DB_PATH = "database/vibejudge.db"

class UserManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_users_table()

    def _init_users_table(self):
        """Create users table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    full_name TEXT,
                    is_active INTEGER DEFAULT 1,
                    is_admin INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            """)
            conn.commit()
        print("✅ Users table ready.")

    def create_user(self, username: str, email: str, password: str,
                    full_name: str = "", is_admin: bool = False) -> dict:
        """Register a new user."""
        user_id = str(uuid.uuid4())
        hashed = hash_password(password)
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT INTO users (id, username, email, hashed_password,
                                      full_name, is_active, is_admin, created_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                """, (user_id, username, email, hashed, full_name, int(is_admin), now))
                conn.commit()
                return {"success": True, "user_id": user_id, "username": username}
            except sqlite3.IntegrityError as e:
                return {"success": False, "error": str(e)}

    def authenticate_user(self, username: str, password: str) -> dict:
        """Authenticate user credentials."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? AND is_active = 1",
                (username,)
            ).fetchone()

        if not row:
            return {"success": False, "error": "User not found"}

        if not verify_password(password, row["hashed_password"]):
            return {"success": False, "error": "Invalid password"}

        # Update last login
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(), row["id"])
            )
            conn.commit()

        return {
            "success": True,
            "user_id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "is_admin": bool(row["is_admin"])
        }

    def get_user_by_id(self, user_id: str) -> dict:
        """Fetch user by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, username, email, full_name, is_admin, created_at, last_login "
                "FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list:
        """List all users (admin only)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, username, email, full_name, is_admin, created_at, last_login "
                "FROM users WHERE is_active = 1 ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
