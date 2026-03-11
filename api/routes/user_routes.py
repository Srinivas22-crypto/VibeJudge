from fastapi import APIRouter, HTTPException
from auth.user_manager import UserManager
from auth.session_manager import session_manager
from api.schemas.analysis_schema import LoginRequest, LoginResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])
user_mgr = UserManager()

@router.post("/register")
async def register(username: str, email: str, password: str, full_name: str = ""):
    """Register a new VibeJudge user."""
    result = user_mgr.create_user(username, email, password, full_name)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"message": "User registered successfully", "user_id": result["user_id"]}

@router.post("/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """Login and receive JWT access token."""
    result = user_mgr.authenticate_user(credentials.username, credentials.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])

    token = session_manager.create_access_token(
        data={"sub": result["user_id"], "username": result["username"]}
    )
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        username=result["username"],
        expires_in=60 * 24 * 60  # seconds
    )

@router.get("/me")
async def get_current_user(token: str):
    """Get current user details from token."""
    result = session_manager.verify_token(token)
    if not result["valid"]:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = result["payload"]["sub"]
    user = user_mgr.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
