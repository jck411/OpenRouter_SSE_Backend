#!/bin/bash
# Simple script to run the OpenRouter SSE Backend server

cd "$(dirname "$0")"

echo "🚀 Starting OpenRouter SSE Backend..."
echo "📁 Using virtual environment: backend/"

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" != *"backend"* ]]; then
    echo "🔧 Activating virtual environment..."
    source backend/bin/activate
fi

# Run the server
backend/bin/openrouter-server
