#!/bin/bash
# Simple script to run the OpenRouter SSE Backend server

cd "$(dirname "$0")"

echo "🚀 Starting OpenRouter SSE Backend..."
echo "📁 Using virtual environment: .venv/"

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" != *".venv"* ]]; then
    echo "🔧 Activating virtual environment..."
    source .venv/bin/activate
fi

# Run the server
.venv/bin/openrouter-server
