from fastapi import APIRouter, HTTPException, Depends
from app.models import BatchPredictionResponse, IrisBatchFeatures, IrisFeatures, ModelInfo, PredictionResponse
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
        "name": "Iris Inference API",
        "version": "1.0.0",
        "description": "FastAPI service for iris flower species inference using scikit-learn's RandomForest classifier",
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

@router.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures, service: ModelService = Depends(get_model_service)):
    try:
        result = service.predict_single(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: IrisBatchFeatures, service: ModelService = Depends(get_model_service)):
    try:
        results = service.predict_batch(batch)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
