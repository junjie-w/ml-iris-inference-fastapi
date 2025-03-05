from pydantic import BaseModel, Field
from typing import List

class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    classes: List[str]
    parameters: dict

class IrisFeatures(BaseModel):
    sepal_length: float = Field(description="Sepal length in cm", json_schema_extra={"example": 5.1})
    sepal_width: float = Field(description="Sepal width in cm", json_schema_extra={"example": 3.5})
    petal_length: float = Field(description="Petal length in cm", json_schema_extra={"example": 1.4})
    petal_width: float = Field(description="Petal width in cm", json_schema_extra={"example": 0.2})

class IrisBatchFeatures(BaseModel):
    samples: List[IrisFeatures]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    features: IrisFeatures

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
