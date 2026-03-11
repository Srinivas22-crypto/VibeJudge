from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.upload_routes import router as upload_router
from api.routes.analysis_routes import router as analysis_router
from api.routes.user_routes import router as user_router

app = FastAPI(
    title="VibeJudge API",
    description="Batch podcast sentiment, bias & emotion analysis API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS (allow frontend dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(upload_router, prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")
app.include_router(user_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "app": "VibeJudge API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs"
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "vibejudge-api"}

if __name__ == "__main__":
    import uvicorn
    from config.api_config import api_config
    uvicorn.run("api.main_api:app", host=api_config.API_HOST,
                port=api_config.API_PORT, reload=True)
