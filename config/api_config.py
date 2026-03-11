import os
from dotenv import load_dotenv

load_dotenv()

class APIConfig:
    # Server settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_VERSION: str = "v1"
    API_PREFIX: str = f"/api/{API_VERSION}"

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "vibejudge-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 30
    RATE_LIMIT_PER_HOUR: int = 200

    # File upload limits
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: list = ["mp3", "wav", "m4a", "ogg", "flac"]

    # Cache settings
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 100

api_config = APIConfig()
