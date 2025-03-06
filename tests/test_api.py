from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data
    assert data["name"] == "Iris Inference API"
    assert "/predict" in data["endpoints"]
    assert "/model/info" in data["endpoints"]

def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_type" in data
    assert "features" in data
    assert "classes" in data
    assert "parameters" in data

    assert data["model_type"] == "RandomForestClassifier"
    assert len(data["features"]) == 4
    assert "sepal_length" in data["features"]
    assert len(data["classes"]) == 3
    assert "setosa" in data["classes"]

def test_predict_endpoint():
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "features" in data
    assert data["features"] == test_data
    
    assert data["prediction"] in ["setosa", "versicolor", "virginica"]
    assert isinstance(data["probability"], float)
    assert 0 <= data["probability"] <= 1

def test_predict_batch_endpoint():
    test_batch = {
        "samples": [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.7,
                "sepal_width": 3.0,
                "petal_length": 5.2,
                "petal_width": 2.3
            }
        ]
    }
    
    response = client.post("/predict/batch", json=test_batch)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) == 2
    
    for i, result in enumerate(data["results"]):
        assert "prediction" in result
        assert "probability" in result
        assert "features" in result
        assert result["features"] == test_batch["samples"][i]
        assert result["prediction"] in ["setosa", "versicolor", "virginica"]
        assert isinstance(result["probability"], float)
        assert 0 <= result["probability"] <= 1
