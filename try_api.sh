#!/bin/bash

# Make sure the API is running: make run

API_URL="http://127.0.0.1:8000"
echo "Trying Iris Inference API at $API_URL"

# Root endpoint
echo -e "\n--- Trying root endpoint ---"
curl -s $API_URL/

# Model info endpoint
echo -e "\n\n--- Trying model info endpoint ---"
curl -s $API_URL/model/info

# Single prediction
echo -e "\n\n--- Trying single prediction ---"
curl -s -X POST \
  $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Batch prediction
echo -e "\n\n--- Trying batch prediction ---"
curl -s -X POST \
  $API_URL/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
      {"sepal_length": 6.2, "sepal_width": 2.9, "petal_length": 4.3, "petal_width": 1.3},
      {"sepal_length": 7.9, "sepal_width": 3.8, "petal_length": 6.4, "petal_width": 2.0}
    ]
  }'

echo -e "\n\nAPI works:) 🪻"
