.PHONY: run dev test coverage train docker-build docker-run docker-pull-remote docker-run-remote

run:
	python run.py

dev:
	uvicorn app.main:app --reload

test:
	pytest

coverage:
	pytest --cov=app tests/

train:
	python model_training.py

docker-build:
	docker build -t iris-inference-api .

docker-run:
	docker run -p 8000:8000 -v $(PWD):/app iris-inference-api

docker-pull-remote:
	docker pull junjiewu0/iris-inference-api || docker pull --platform linux/amd64 junjiewu0/iris-inference-api

docker-run-remote:
	docker run -p 8000:8000 junjiewu0/iris-inference-api || docker run --platform linux/amd64 -p 8000:8000 junjiewu0/iris-inference-api
