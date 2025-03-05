import pickle

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
    