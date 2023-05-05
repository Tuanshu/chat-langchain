.PHONY: start
start:
	uvicorn main:app --reload --port 1234

.PHONY: format
format:
	black .
	isort .