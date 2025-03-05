import pickle
import numpy as np
from app.models import IrisBatchFeatures, IrisFeatures

class ModelService:
    def __init__(self, model_path="iris_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()

        self.iris_species = {
            0: "setosa",
            1: "versicolor",
            2: "virginica"
        }

    def load_model(self):
        """Load the machine learning model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def get_model_info(self):
        """Get information about the model."""
        if self.model is None:
            return None
        
        return {
            "model_type": type(self.model).__name__,
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "classes": list(self.iris_species.values()),
            "parameters": self.model.get_params()
        }
    
    def predict_single(self, features: IrisFeatures):
        """Make a prediction for a single sample."""
        if self.model is None:
            return None
        
        feature_array = np.array([
            [features.sepal_length, 
            features.sepal_width, 
            features.petal_length, 
            features.petal_width]
        ])
        
        prediction = self.model.predict(feature_array)[0]
        probability = self.model.predict_proba(feature_array)[0][prediction].round(3)
        
        return {
            "prediction": self.iris_species[prediction],
            "probability": float(probability),
            "features": features
        }

    def predict_batch(self, batch: IrisBatchFeatures):
        """Make predictions for multiple samples."""
        if self.model is None:
            return None
        
        features_list = []
        for sample in batch.samples:
            features_list.append([
                sample.sepal_length,
                sample.sepal_width,
                sample.petal_length,
                sample.petal_width
            ])
        
        feature_array = np.array(features_list)
        
        predictions = self.model.predict(feature_array)
        probabilities = self.model.predict_proba(feature_array)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "prediction": self.iris_species[pred],
                "probability": float(probabilities[i][pred].round(3)),
                "features": batch.samples[i]
            })
        
        return {"results": results}
