# OpenRouter SSE Backend - Development Environment Setup
# Source this file to set up your development environment

# Add this to your ~/.bashrc or ~/.zshrc for permanent setup:
# alias openrouter-dev='cd /home/jack/REPOS/OpenRouter_SSE_Backend && source ./dev-env.sh'

# Or run: source ./dev-env.sh

echo "🚀 Setting up OpenRouter SSE Backend development environment..."

# Navigate to project root
cd "$(dirname "${BASH_SOURCE[0]}")"

# Show current location
echo "📁 Project directory: $(pwd)"

# Activate virtual environment
if [[ "$VIRTUAL_ENV" != *"backend"* ]]; then
    echo "🔧 Activating virtual environment..."
    source backend/bin/activate
else
    echo "✅ Virtual environment already activated"
fi

# Show Python info
echo "🐍 Python: $(backend/bin/python --version) at backend/bin/python"

# Show available commands
echo ""
echo "🛠️  Available commands:"
echo "   ./run_server.sh           - Start the FastAPI server"
echo "   backend/bin/openrouter-server - Start server directly"
echo "   pytest tests/             - Run tests"
echo "   deactivate                - Exit virtual environment"
echo ""
