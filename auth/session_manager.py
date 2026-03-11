from datetime import datetime, timedelta
from jose import JWTError, jwt
from config.api_config import api_config

class SessionManager:
    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (
            expires_delta or timedelta(minutes=api_config.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, api_config.SECRET_KEY, algorithm=api_config.ALGORITHM)

    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, api_config.SECRET_KEY,
                                  algorithms=[api_config.ALGORITHM])
            return {"valid": True, "payload": payload}
        except JWTError as e:
            return {"valid": False, "error": str(e)}

session_manager = SessionManager()
