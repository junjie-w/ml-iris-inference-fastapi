from fastapi import APIRouter, HTTPException, Depends
from app.models import ModelInfo
from app.service import ModelService

router = APIRouter()

def get_model_service():
    service = ModelService()
    if service.model is None:
        from model_training import train_model
        try:
            train_model()
            service = ModelService()
        except Exception as e:
            raise HTTPException(
                status_code=503, 
                detail=f"Model is not available and could not be trained: {str(e)}"
            )
    return service

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


@router.get("/model/info", response_model=ModelInfo)
def model_info(service: ModelService = Depends(get_model_service)):
    return service.get_model_info()