import pickle
from app.service import ModelService
from sklearn.ensemble import RandomForestClassifier

class TestModelService:
    def test_load_model_success(self, tmp_path):
        test_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(test_model, f)
        
        service = ModelService(model_path=str(model_path))
        
        assert service.model is not None
        
        info = service.get_model_info()
        assert info["features"] == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        assert info["classes"] == ["setosa", "versicolor", "virginica"]
        assert "n_estimators" in info["parameters"]
        assert info["parameters"]["n_estimators"] == 100

    def test_load_model_failure(self):
        service = ModelService(model_path="nonexistent_model.pkl")
        
        assert service.model is None
        assert service.get_model_info() is None
    
    def test_get_model_info(self, tmp_path):
        test_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(test_model, f)
        
        service = ModelService(model_path=str(model_path))
        
        info = service.get_model_info()
        assert info is not None
        assert "model_type" in info
        assert "features" in info
        assert "classes" in info
        assert "parameters" in info
        assert len(info["features"]) == 4
        assert len(info["classes"]) == 3
