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

    def test_predict_single(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        test_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        import numpy as np
        X = np.array([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]])
        y = np.array([0, 1, 2])
        test_model.fit(X, y)
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(test_model, f)
        
        service = ModelService(model_path=str(model_path))
        
        from app.models import IrisFeatures
        test_features = IrisFeatures(
            sepal_length=5.1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2
        )
        
        result = service.predict_single(test_features)
        
        assert result is not None
        assert "prediction" in result
        assert "probability" in result
        assert "features" in result
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]
        assert isinstance(result["probability"], float)
        assert 0 <= result["probability"] <= 1

    def test_predict_batch(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier
        test_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        import numpy as np
        X = np.array([[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]])
        y = np.array([0, 1, 2])
        test_model.fit(X, y)
        
        model_path = tmp_path / "test_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(test_model, f)
        
        service = ModelService(model_path=str(model_path))
        
        from app.models import IrisFeatures, IrisBatchFeatures
        
        test_batch = IrisBatchFeatures(samples=[
            IrisFeatures(
                sepal_length=5.1,
                sepal_width=3.5,
                petal_length=1.4,
                petal_width=0.2
            ),
            IrisFeatures(
                sepal_length=6.7,
                sepal_width=3.0,
                petal_length=5.2,
                petal_width=2.3
            )
        ])
        
        result = service.predict_batch(test_batch)
        
        assert result is not None
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 2
        
        for i, pred_result in enumerate(result["results"]):
            assert "prediction" in pred_result
            assert "probability" in pred_result
            assert "features" in pred_result
            assert pred_result["prediction"] in ["setosa", "versicolor", "virginica"]
            assert isinstance(pred_result["probability"], float)
            assert 0 <= pred_result["probability"] <= 1
            assert pred_result["features"] == test_batch.samples[i]
