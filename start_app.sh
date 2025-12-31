#!/usr/bin/env bash
################################################################################
# Washington DC Weather Dashboard - Launcher (Linux/Mac)
#
# This script runs the orchestrator which automates:
#   1. Dependency checking
#   2. Weather forecast generation
#   3. Dashboard launch
#
# Usage:
#   ./start_app.sh                  - Full workflow (with forecast generation)
#   ./start_app.sh --skip-forecast  - Skip forecast, use existing data
#   ./start_app.sh --port 8502      - Use custom port
#
# Make executable with: chmod +x start_app.sh
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Print banner
echo ""
echo "================================================================================"
echo "  Washington DC Weather Dashboard - Launcher"
echo "================================================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    echo ""
    echo "Please install Python 3.8 or higher:"
    echo "  - Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "  - macOS: brew install python3"
    echo "  - Or download from: https://www.python.org/downloads/"
    echo ""
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python version: $PYTHON_VERSION"

# Check if run_weather_dashboard.py exists
if [ ! -f "run_weather_dashboard.py" ]; then
    print_error "run_weather_dashboard.py not found!"
    echo ""
    echo "Make sure you're running this script from the project directory:"
    echo "  $(pwd)"
    echo ""
    exit 1
fi

print_info "Launching Weather Dashboard Orchestrator..."
echo ""

# Run the orchestrator with any command-line arguments
python3 run_weather_dashboard.py "$@"
EXIT_CODE=$?

# Check exit code
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "  Dashboard Launch Failed"
    echo "================================================================================"
    echo ""
    print_error "The orchestrator encountered an error (exit code: $EXIT_CODE)"
    echo ""
    echo "Common issues:"
    echo "  - Missing dependencies: Run 'pip3 install -r requirements.txt'"
    echo "  - Port already in use: Try './start_app.sh --port 8502'"
    echo "  - Python version too old: Need Python 3.8+"
    echo ""
    exit $EXIT_CODE
fi

# Normal exit
echo ""
echo "================================================================================"
echo "  Dashboard Stopped"
echo "================================================================================"
echo ""
print_success "Thank you for using Weather Dashboard!"
echo ""
echo "To run again:"
echo "  ./start_app.sh"
echo ""
exit 0
