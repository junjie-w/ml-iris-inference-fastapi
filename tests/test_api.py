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
