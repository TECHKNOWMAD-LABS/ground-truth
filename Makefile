.PHONY: test lint format security clean install

VENV := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

install:
	uv venv $(VENV) --python 3.12
	uv pip install -e ".[dev]" pytest pytest-cov hypothesis ruff mypy

test:
	$(PYTEST) -v --tb=short --cov=groundtruth --cov-report=term-missing

test-fast:
	$(PYTEST) -v --tb=short -x

test-property:
	$(PYTEST) tests/test_property_based.py -v --tb=short

lint:
	$(RUFF) check src/ tests/

format:
	$(RUFF) format src/ tests/
	$(RUFF) check src/ tests/ --fix

security:
	@echo "Running security scan..."
	@grep -rn "api_key\|API_KEY\|password\|secret\|sk-\|pk-\|-----BEGIN\|hardcoded" src/ tests/ --include="*.py" \
		|| echo "No obvious secrets found"
	@echo "Checking for subprocess/eval/exec usage..."
	@grep -rn "subprocess\|eval(\|exec(" src/ --include="*.py" \
		|| echo "No subprocess/eval/exec found"
	@echo "Security scan complete."

clean:
	rm -rf .venv .pytest_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

coverage-html:
	$(PYTEST) --cov=groundtruth --cov-report=html
	@echo "Coverage report at htmlcov/index.html"

examples:
	$(PYTHON) examples/basic_detection.py
	$(PYTHON) examples/batch_detection.py
	$(PYTHON) examples/custom_detector.py
