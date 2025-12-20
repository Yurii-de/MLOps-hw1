IMAGE_NAME = my-ml-service
TAG = latest

.PHONY: build-push test lint

build-push:
	docker build -t $(IMAGE_NAME):$(TAG) .
	docker push $(IMAGE_NAME):$(TAG)

test:
	uv run pytest -v tests/

lint:
	uv run ruff check .
