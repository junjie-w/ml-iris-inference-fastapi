#!/bin/bash

# Make sure the API is running: make run

API_URL="http://127.0.0.1:8000"
echo "Testing Iris Inference API at $API_URL"

# Test root endpoint
echo -e "\n--- Testing root endpoint ---"
curl -s $API_URL/

# Test model info endpoint
echo -e "\n\n--- Testing model info endpoint ---"
curl -s $API_URL/model/info

# Test single prediction
echo -e "\n\n--- Testing single prediction ---"
curl -s -X POST \
  $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Test batch prediction
echo -e "\n\n--- Testing batch prediction ---"
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

echo -e "\n\nAPI testing completed"
