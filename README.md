# Iris Inference API 🪻

![Python Version](https://img.shields.io/badge/python-3.13.2-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.11-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![Docker](https://img.shields.io/badge/Docker-enabled-blue)

A [FastAPI](https://fastapi.tiangolo.com/) service for predicting iris flower species using [scikit-learn](https://scikit-learn.org/)'s [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

The model is trained on the classic [Iris flower dataset](https://archive.ics.uci.edu/dataset/53/iris). By providing flower measurements (sepal and petal dimensions) as input, the API returns the most likely species classification with probability score.

Available as a [Docker image](https://hub.docker.com/r/junjiewu0/iris-inference-api).

## 🪻 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |
| `/model/info` | GET | Classifier details |
| `/predict` | POST | Single iris flower prediction |
| `/predict/batch` | POST | Batch iris flower predictions |

**Try all API endpoints with the included script:**

```bash
./try_api.sh
```

<details>
<summary>Example API Requests</summary>

### Root Endpoint

```bash
curl http://localhost:8000/
```

### Model Info Endpoint

```bash
curl http://localhost:8000/model/info
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
      },
      {
        "sepal_length": 6.2,
        "sepal_width": 2.9,
        "petal_length": 4.3,
        "petal_width": 1.3
      }
    ]
  }'
```
</details>

## 🪻 Development Setup

```bash
# Clone repo
git clone https://github.com/junjie-w/ml-iris-inference-fastapi.git
cd ml-iris-inference-fastapi

# Install dependencies
pip install -r requirements.txt

# Train model (creates iris_model.pkl)
python model_training.py

# Run the API
python run.py
```

- API base URL: http://localhost:8000
- Interactive OpenAPI documentation: http://localhost:8000/docs
- OpenAPI specification (JSON): http://localhost:8000/openapi.json

## 🪻 Docker Usage

### Pre-built Image from Docker Hub

```bash
# Pull image from Docker Hub
docker pull junjiewu0/iris-inference-api

# For ARM-based machines (Apple Silicon, etc.)
docker pull --platform linux/amd64 junjiewu0/iris-inference-api

# Run container
docker run -p 8000:8000 junjiewu0/iris-inference-api

# For ARM-based machines (Apple Silicon, etc.)
docker run --platform linux/amd64 -p 8000:8000 junjiewu0/iris-inference-api
```

### Build Image Locally

```bash
# Build image
docker build -t iris-inference-api .

# Run container
docker run -p 8000:8000 iris-inference-api
```

## 🪻 Run Tests

```bash
# Run the test suite
pytest

# For test coverage
pytest --cov=app tests/
```

## 🪻 Makefile Commands

```bash
make run                     # Start the API server
make dev                     # Start the server with auto-reload
make test                    # Run tests
make coverage                # Run tests with coverage report
make train                   # Train the model (creates iris_model.pkl)
make docker-build            # Build the Docker image
make docker-run              # Run container from local image
make docker-pull-remote      # Pull pre-built image from Docker Hub
make docker-run-remote       # Run container from pre-built Docker Hub image
```
