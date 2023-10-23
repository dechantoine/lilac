#!/bin/bash

# Fail if any of the commands below fail.
set -e

# Activate the current py virtual env.
case "$OSTYPE" in
  linux*)
    echo "OSTYPE : LINUX"
    source $(poetry env info --path)/bin/activate ;;
  msys*)
    echo "OSTYPE : WINDOWS"
    source $(poetry env info --path)/Scripts/activate ;;
esac

echo "Linting python with ruff..."
poetry run ruff lilac

poetry run isort --check lilac/

echo "Checking python types with mypy..."
poetry run mypy lilac
