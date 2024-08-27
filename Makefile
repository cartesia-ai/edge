install-dev:
	pip install -r requirements-dev.txt

# Pytype can be quite expensive to run.
# TODO: Determine if we want to use a lazier type checker.
# cd cartesia-mlx && pytype --disable=import-error . && cd .. 
# cd cartesia-metal && pytype --disable=import-error . && cd .. 
# cd cartesia-pytorch && pytype --disable=import-error . && cd .. 
lint:
	isort -c .
	ruff check .
	ruff format --check .


autoformat:
	isort --atomic .
	ruff format .
