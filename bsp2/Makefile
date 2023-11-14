# Makefile

# Define the Python interpreter
PYTHON=python3

# Define the MyPy command
MYPY=mypy

# Define the test command for unittest
TEST=unittest

# Default target executed when no arguments are given to make.
default: clean test mypy clean_mypy_cache

# Target for running tests
test: clean
	@$(PYTHON) -m $(TEST) discover -s test -p 'test_*.py'

# Target for running MyPy
mypy: clean
	@$(MYPY) src/
	@make clean_mypy_cache


# Target for cleaning up
clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete

clean_mypy_cache:
	@echo "Cleaning up MyPy cache..."
	@rm -rf .mypy_cache

.PHONY: default test mypy clean
