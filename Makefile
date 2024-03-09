#!/bin/bash

# Define macros for directories
SRC = src
DATA = data
DOCS = docs

.PHONY: all clean run

# Default target executed
all: run

# Run the main application
run:
    @echo "Running the main application..."
    python main.py

# Install dependencies
install:
    @echo "Installing dependencies..."
    pip install -r requirements.txt

# Clean up pycache and temporary files
clean:
    @echo "Cleaning up..."
    find . -type f -name '*.pyc' -delete
    find . -type d -name '__pycache__' -exec rm -r {} +
    find . -type f -name '*.log' -delete
    find . -type f -name '.DS_Store' -delete
    find . -type f -name '*~' -delete
