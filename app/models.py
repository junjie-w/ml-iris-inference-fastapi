from pydantic import BaseModel
from typing import List

class ModelInfo(BaseModel):
    model_type: str
    features: List[str]
    classes: List[str]
    parameters: dict