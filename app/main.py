from fastapi import FastAPI
from app.router import router

app = FastAPI(
    title="Iris Flower Inference API", 
    description="FastAPI service for iris flower species inference using a RandomForest classifier",
    version="1.0.0"
)

app.include_router(router)
