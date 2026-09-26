from fastapi import FastAPI

from backend.routes import heart_router, diabetes_router, stroke_router

app = FastAPI(
    title="AI Multi-Disease Risk Screening API",
    description="Screening-only API for heart disease, diabetes, and stroke.",
    version="1.0.0",
)

app.include_router(heart_router)
app.include_router(diabetes_router)
app.include_router(stroke_router)


@app.get("/")
def root():
    return {
        "project": "AI Multi-Disease Risk Screening System",
        "status": "running",
        "screening_only": True,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }
