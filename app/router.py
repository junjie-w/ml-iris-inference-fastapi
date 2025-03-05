from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():
    return {
        "name": "Iris Flower Inference API",
        "version": "1.0.0",
        "description": "FastAPI service for iris flower species inference using a RandomForest classifier",
        "endpoints": {
            "/predict": "Make a prediction for a single iris sample",
            "/predict/batch": "Make predictions for multiple iris samples",
            "/model/info": "Get information about the trained model"
        },
        "docs_url": "/docs",
        "author": "Junjie Wu"
    }
