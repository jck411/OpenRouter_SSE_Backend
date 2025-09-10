#!/bin/bash
# Script to activate the OpenRouter SSE Backend virtual environment

cd "$(dirname "$0")"

echo "🔧 Activating OpenRouter SSE Backend virtual environment..."
echo "📁 Environment location: $(pwd)/backend/"
echo "🐍 Python version: $(backend/bin/python --version)"
echo ""
echo "To deactivate later, run: deactivate"
echo "=================================================="

# Activate the virtual environment
source backend/bin/activate
