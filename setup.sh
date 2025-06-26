#!/bin/bash
# Quick Start Script - One-click setup for development environment

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Cats & Dogs Image Classification System - Quick Start"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version is too low, 3.9 or higher is required"
    echo "Current version: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check and install pipenv
if ! command -v pipenv &> /dev/null; then
    echo "🔧 Installing Pipenv..."
    pip3 install pipenv
fi

echo "✅ Pipenv is installed"

# Install dependencies
echo "📦 Installing project dependencies..."
pipenv install

# Install development dependencies
echo "🛠️  Installing development dependencies..."
pipenv install --dev

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
pipenv run python -m ipykernel install --user --name=tensorflow-cats-dogs --display-name="TensorFlow Cats & Dogs"

# Run environment check
echo "🔍 Running environment check..."
pipenv run python setup_env.py

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run 'pipenv shell' to enter the virtual environment"
echo "2. Run 'make data' to prepare the dataset"
echo "3. Run 'make train' to train the model"
echo "4. Run 'make web' to start the web application"
echo ""
echo "Or use 'make help' to see all available commands"
