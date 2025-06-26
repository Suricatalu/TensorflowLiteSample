# Dog and Cat Image Classification System Makefile (Pipenv Version)
# Use make <command> to execute various operations

.PHONY: help setup install data train predict web clean dev-install shell
.PHONY: tflite-predict

# Default target
help:
	@echo "Dog and Cat Image Classification System - Available Commands (Pipenv):"
	@echo ""
	@echo "  setup       - Environment setup and checks"
	@echo "  install     - Install all dependencies"
	@echo "  dev-install - Install development dependencies"
	@echo "  shell       - Enter Pipenv virtual environment"
	@echo "  data        - Download and prepare dataset"
	@echo "  train       - Train the model"
	@echo "  predict     - Predict example (requires image path)"
	@echo "  web         - Start web application"
	@echo "  format      - Format code"
	@echo "  lint        - Code linting"
	@echo "  clean       - Clean temporary files"
	@echo "  tflite-predict - Run TFLite inference (requires image path)"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make install"
	@echo "  make shell"
	@echo "  make data"
	@echo "  make train"
	@echo "  make predict IMG=path/to/image.jpg"
	@echo "  make web"

# Environment setup
setup:
	@echo "🚀 Starting environment setup..."
	python setup_env.py

# Install production dependencies
install:
	@echo "📦 Installing dependencies..."
	pipenv install

# Install development dependencies
dev-install:
	@echo "🛠️  Installing development dependencies..."
	pipenv install --dev

# Enter virtual environment
shell:
	@echo "🐚 Entering Pipenv virtual environment..."
	pipenv shell

# Prepare dataset
data:
	@echo "📁 Preparing dataset..."
	pipenv run prepare-data

# Train the model
train:
	@echo "🧠 Starting model training..."
	pipenv run train

# Predict (Usage: make predict IMG=path/to/image.jpg)
predict:
	@if [ -z "$(IMG)" ]; then \
		echo "❌ Please provide an image path: make predict IMG=path/to/image.jpg"; \
	else \
		echo "🔍 Predicting image: $(IMG)"; \
		pipenv run python predict.py $(IMG); \
	fi

# Start web application
web:
	@echo "🌐 Starting web application..."
	@echo "Browser will automatically open http://localhost:5000"
	pipenv run web

# Format code
format:
	@echo "🎨 Formatting code..."
	pipenv run format

# Code linting
lint:
	@echo "🔍 Code linting..."
	pipenv run lint

# Clean temporary files
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	rm -f *.png
	rm -f *.jpg
	rm -f *.jpeg
	@echo "✅ Cleaning completed"

# TFLite inference (Usage: make tflite-predict IMG=path/to/image.jpg)
tflite-predict:
	@if [ -z "$(IMG)" ]; then \
		echo "❌ Please provide an image path: make tflite-predict IMG=path/to/image.jpg"; \
	else \
		echo "🔍 Running TFLite inference on: $(IMG)"; \
		pipenv run python tflite_predict.py --model models_tflite/cat_dog_classifier.tflite --image $(IMG); \
	fi

# Full workflow (from scratch)
all: setup install data train
	@echo "🎉 Full workflow completed!"
	@echo "You can now use 'make predict IMG=image_path' or 'make web' to test the system"

# Quick test
test:
	@echo "🧪 Running quick test..."
	pipenv run python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"
	@if [ -d "models/cat_dog_classifier" ]; then \
		echo "✅ Model file exists"; \
	else \
		echo "❌ Model file does not exist, please run 'make train' first"; \
	fi

# Display system information
info:
	@echo "📊 System Information:"
	@echo "Python Version: $(shell python --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Pipenv Version: $(shell pipenv --version 2>/dev/null || echo 'Not Installed')"
	@echo "Virtual Environment: $(shell pipenv --venv 2>/dev/null || echo 'Not Created')"
	@echo "TensorFlow: $(shell pipenv run python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not Installed')"

# Display dependency tree
deps:
	@echo "📋 Dependency Tree:"
	pipenv graph

# Security check
security:
	@echo "🔒 Security Check..."
	pipenv check

# Update dependencies
update:
	@echo "⬆️  Updating dependencies..."
	pipenv update

# Create Jupyter kernel
jupyter-kernel:
	@echo "📓 Creating Jupyter kernel..."
	pipenv run python -m ipykernel install --user --name=tensorflow-cats-dogs
