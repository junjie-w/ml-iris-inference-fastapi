.PHONY: run dev test coverage train docker-build docker-run

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
