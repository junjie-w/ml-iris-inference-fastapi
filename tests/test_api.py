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
    assert data["name"] == "Iris Flower Inference API"
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
