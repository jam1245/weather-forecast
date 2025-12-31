#!/usr/bin/env bash
################################################################################
# Washington DC Weather Dashboard - Quick Start (Linux/Mac)
#
# This script launches the dashboard WITHOUT generating new forecasts.
# It uses existing data files for faster startup.
#
# Perfect for:
#   - Quick testing during development
#   - Viewing existing data without API calls
#   - Faster launches when data is already fresh
#
# Usage:
#   ./quick_start.sh              - Launch with existing data
#   ./quick_start.sh --port 8502  - Use custom port
#
# For full workflow with forecast generation, use ./start_app.sh instead
#
# Make executable with: chmod +x quick_start.sh
#
################################################################################

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
echo "  Weather Dashboard - QUICK START (Skip Forecast Generation)"
echo "================================================================================"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    echo ""
    echo "Please install Python 3.8 or higher"
    echo ""
    exit 1
fi

# Check if run_weather_dashboard.py exists
if [ ! -f "run_weather_dashboard.py" ]; then
    print_error "run_weather_dashboard.py not found!"
    echo ""
    echo "Make sure you're running this script from the project directory."
    echo ""
    exit 1
fi

# Check if data files exist
if [ ! -f "weather_historical_forecast.csv" ]; then
    print_warning "weather_historical_forecast.csv not found!"
    echo ""
    echo "You need to generate data first by running:"
    echo "  ./start_app.sh"
    echo ""
    echo "Or manually run:"
    echo "  python3 weather_forecast.py"
    echo ""
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo ""
        echo "Cancelled. Run ./start_app.sh to generate data first."
        exit 1
    fi
fi

print_info "Launching dashboard with --skip-forecast..."
print_info "Using existing data (no forecast generation)"
echo ""

# Run orchestrator with --skip-forecast flag and pass any additional arguments
python3 run_weather_dashboard.py --skip-forecast "$@"
EXIT_CODE=$?

# Check if orchestrator failed
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "  Quick Start Failed"
    echo "================================================================================"
    echo ""
    print_error "Check the output above for error details."
    echo ""
    echo "Try running full workflow:"
    echo "  ./start_app.sh"
    echo ""
    exit $EXIT_CODE
fi

# Normal exit
echo ""
echo "================================================================================"
echo "  Dashboard Stopped"
echo "================================================================================"
echo ""
print_success "To run again:"
echo "  ./quick_start.sh       - Quick start (skip forecast)"
echo "  ./start_app.sh         - Full workflow (with forecast)"
echo ""
exit 0
