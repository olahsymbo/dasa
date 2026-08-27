TARGET_DIR = target
PATHS = src

install:
	@poetry install

add:
	@poetry add $(package)

add-dev:
	@poetry add --dev $(package)

remove:
	@poetry remove $(package)

lint:
	@poetry run flake8 $(PATHS)
	@poetry run black --check ${PATHS}

lint-fix:
	@poetry run isort ${PATHS}
	@poetry run black ${PATHS}

type-check:
	@poetry run mypy $(PATHS)

format:
	@poetry run isort $(PATHS)
	@poetry run black $(PATHS)

check: lint type-check test

test:
	@poetry run pytest

dist: check
	@poetry build

update:
	@poetry update

clean:
	@find $(TARGET_DIR) -type f -name "*.pyc" -delete
	@find $(TARGET_DIR) -type f -name "*.pyo" -delete