#!/usr/bin/env python3
"""
Weather Dashboard Orchestrator

This script automates the complete workflow for the Washington DC Weather Dashboard:
1. Checks dependencies
2. Generates weather forecasts (API + ML)
3. Verifies output files
4. Launches Streamlit dashboard

Usage:
    python run_weather_dashboard.py
    python run_weather_dashboard.py --skip-forecast
    python run_weather_dashboard.py --force-retrain --port 8502

Author: Weather Dashboard Team
Version: 1.0.0
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
from datetime import datetime
import time
import importlib.util

# Try to import colorama for colored output
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    # Fallback: no colors
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    class Back:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


# Script configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
REQUIRED_FILES = {
    'weather_forecast.py': 'Weather forecast generation script',
    'weather_app.py': 'Streamlit dashboard application',
    'requirements.txt': 'Python dependencies list'
}

OUTPUT_FILES = {
    'weather_historical_forecast.csv': 'Combined weather data',
    'temperature_historical_forecast.png': 'Weather visualization',
}

OPTIONAL_OUTPUTS = {
    'models_cache/': 'ML model cache directory'
}

# Core dependencies (required)
CORE_DEPENDENCIES = [
    'streamlit',
    'pandas',
    'matplotlib',
    'requests',
    'numpy'
]

# ML dependencies (optional)
ML_DEPENDENCIES = [
    'prophet',
    'statsmodels',
    'pmdarima',
    'sklearn',
    'joblib'
]


def print_header(message, char='='):
    """Print a formatted header"""
    width = 80
    print()
    print(Fore.CYAN + Style.BRIGHT + char * width)
    print(Fore.CYAN + Style.BRIGHT + f"  {message}")
    print(Fore.CYAN + Style.BRIGHT + char * width)
    print()


def print_success(message):
    """Print success message"""
    print(Fore.GREEN + "✅ " + message)


def print_error(message):
    """Print error message"""
    print(Fore.RED + "❌ " + message)


def print_warning(message):
    """Print warning message"""
    print(Fore.YELLOW + "⚠️  " + message)


def print_info(message):
    """Print info message"""
    print(Fore.BLUE + "ℹ️  " + message)


def print_step(step_num, total_steps, message):
    """Print step progress"""
    print(Fore.MAGENTA + Style.BRIGHT + f"\n[Step {step_num}/{total_steps}] {message}")
    print(Fore.MAGENTA + "-" * 80)


def check_file_exists(filepath, description):
    """Check if a required file exists"""
    path = SCRIPT_DIR / filepath
    if path.exists():
        print_success(f"{description}: {filepath}")
        return True
    else:
        print_error(f"Missing {description}: {filepath}")
        return False


def check_required_files():
    """Check if all required project files exist"""
    print_step(1, 5, "Checking Required Files")

    all_present = True
    for filepath, description in REQUIRED_FILES.items():
        if not check_file_exists(filepath, description):
            all_present = False

    if not all_present:
        print_error("\nMissing required files! Make sure you're in the project directory.")
        return False

    print_success("\nAll required files present!")
    return True


def check_dependency(package_name):
    """Check if a Python package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def check_dependencies():
    """Check if all dependencies are installed"""
    print_step(2, 5, "Checking Dependencies")

    # Check core dependencies
    print(Fore.WHITE + Style.BRIGHT + "\nCore Dependencies:")
    missing_core = []
    for package in CORE_DEPENDENCIES:
        if check_dependency(package):
            print_success(f"{package:20s} - Installed")
        else:
            print_error(f"{package:20s} - Missing")
            missing_core.append(package)

    # Check ML dependencies
    print(Fore.WHITE + Style.BRIGHT + "\nML Dependencies (Optional):")
    missing_ml = []
    for package in ML_DEPENDENCIES:
        # Handle sklearn special case
        check_name = 'sklearn' if package == 'sklearn' else package
        if check_dependency(check_name):
            print_success(f"{package:20s} - Installed")
        else:
            print_warning(f"{package:20s} - Missing (ML forecasting will be disabled)")
            missing_ml.append(package)

    # Handle missing core dependencies
    if missing_core:
        print_error("\n❌ Missing core dependencies!")
        print_info("\nInstall with:")
        print(Fore.WHITE + "    pip install -r requirements.txt")
        print_info("\nOr install core only:")
        print(Fore.WHITE + f"    pip install {' '.join(missing_core)}")
        return False

    # Warn about missing ML dependencies
    if missing_ml:
        print_warning("\n⚠️  Some ML dependencies are missing.")
        print_info("ML forecasting will be disabled. Dashboard will still work with API forecasts only.")
        print_info("\nTo enable ML forecasting, install:")
        print(Fore.WHITE + f"    pip install {' '.join(missing_ml)}")
        print_info("\nOr install all dependencies:")
        print(Fore.WHITE + "    pip install -r requirements.txt")

        response = input(Fore.YELLOW + "\nContinue without ML forecasting? (y/n): ")
        if response.lower() != 'y':
            print_info("Exiting. Install dependencies and try again.")
            return False

    print_success("\n✅ All core dependencies installed!")
    return True


def run_forecast_generation(force_retrain=False):
    """Run the weather forecast generation script"""
    print_step(3, 5, "Generating Weather Forecasts")

    print_info("Running weather_forecast.py...")
    print_info("This will:")
    print("  • Fetch 30 days of historical data from Open-Meteo API")
    print("  • Fetch 7-day forecast from Open-Meteo API")
    print("  • Train ML models (Prophet) if libraries are installed")
    print("  • Generate combined CSV and visualization")

    if force_retrain:
        print_warning("Force retrain enabled - ML models will be retrained from scratch")

    print()

    # Build command
    script_path = SCRIPT_DIR / 'weather_forecast.py'
    cmd = [sys.executable, str(script_path)]

    try:
        # Run the script and show output in real-time
        print(Fore.WHITE + Style.DIM + "-" * 80)

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=SCRIPT_DIR
        )

        # Stream output
        for line in process.stdout:
            print(Fore.WHITE + Style.DIM + line, end='')

        process.wait()

        print(Fore.WHITE + Style.DIM + "-" * 80)
        print()

        if process.returncode == 0:
            print_success("Forecast generation completed successfully!")
            return True
        else:
            print_error(f"Forecast generation failed with exit code {process.returncode}")
            return False

    except FileNotFoundError:
        print_error(f"Python executable not found: {sys.executable}")
        return False
    except Exception as e:
        print_error(f"Error running forecast generation: {e}")
        return False


def verify_outputs():
    """Verify that output files were created"""
    print_step(4, 5, "Verifying Output Files")

    all_present = True

    # Check required outputs
    print(Fore.WHITE + Style.BRIGHT + "Required Outputs:")
    for filepath, description in OUTPUT_FILES.items():
        path = SCRIPT_DIR / filepath
        if path.exists():
            size = path.stat().st_size
            modified = datetime.fromtimestamp(path.stat().st_mtime)
            size_kb = size / 1024
            print_success(f"{description}")
            print(Fore.WHITE + Style.DIM + f"       File: {filepath}")
            print(Fore.WHITE + Style.DIM + f"       Size: {size_kb:.1f} KB")
            print(Fore.WHITE + Style.DIM + f"       Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print_error(f"{description}: {filepath} - NOT FOUND")
            all_present = False

    # Check optional outputs
    print(Fore.WHITE + Style.BRIGHT + "\nOptional Outputs:")
    for filepath, description in OPTIONAL_OUTPUTS.items():
        path = SCRIPT_DIR / filepath
        if path.exists():
            if path.is_dir():
                files = list(path.glob('*'))
                print_success(f"{description}")
                print(Fore.WHITE + Style.DIM + f"       Location: {filepath}")
                print(Fore.WHITE + Style.DIM + f"       Files: {len(files)}")
                for f in files:
                    print(Fore.WHITE + Style.DIM + f"         - {f.name}")
            else:
                print_success(f"{description}: {filepath}")
        else:
            print_info(f"{description}: {filepath} - Not present (ML models not cached)")

    if all_present:
        print_success("\n✅ All required outputs verified!")
        return True
    else:
        print_warning("\n⚠️  Some outputs are missing")
        return False


def launch_streamlit(port=8501, extra_args=None):
    """Launch the Streamlit dashboard"""
    print_step(5, 5, "Launching Streamlit Dashboard")

    print_info(f"Starting Streamlit on port {port}...")
    print_info("The dashboard will open automatically in your browser")
    print_info("Press Ctrl+C to stop the server")
    print()

    # Build command
    cmd = [
        sys.executable,
        '-m', 'streamlit',
        'run',
        str(SCRIPT_DIR / 'weather_app.py'),
        '--server.port', str(port)
    ]

    # Add extra arguments if provided
    if extra_args:
        cmd.extend(extra_args)

    try:
        print(Fore.WHITE + Style.DIM + "-" * 80)
        print(Fore.GREEN + Style.BRIGHT + f"\n🚀 Launching Weather Dashboard at http://localhost:{port}\n")
        print(Fore.WHITE + Style.DIM + "-" * 80)
        print()

        # Run Streamlit (this will block until Ctrl+C)
        subprocess.run(cmd, cwd=SCRIPT_DIR)

    except KeyboardInterrupt:
        print()
        print_info("\nStreamlit server stopped by user")
        return True
    except FileNotFoundError:
        print_error("Streamlit not found! Make sure it's installed:")
        print(Fore.WHITE + "    pip install streamlit")
        return False
    except Exception as e:
        print_error(f"Error launching Streamlit: {e}")
        print_info("\nTroubleshooting:")
        print("  • Check if port is already in use: Try --port 8502")
        print("  • Verify Streamlit is installed: pip install streamlit")
        print("  • Try running manually: streamlit run weather_app.py")
        return False


def main():
    """Main orchestrator function"""
    parser = argparse.ArgumentParser(
        description='Weather Dashboard Orchestrator - Automates the complete workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_weather_dashboard.py
  python run_weather_dashboard.py --skip-forecast
  python run_weather_dashboard.py --force-retrain
  python run_weather_dashboard.py --port 8502
  python run_weather_dashboard.py --skip-forecast --port 8502
        """
    )

    parser.add_argument(
        '--skip-forecast',
        action='store_true',
        help='Skip forecast generation and launch dashboard with existing data'
    )

    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force ML model retraining (slower but ensures fresh models)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Streamlit server port (default: 8501)'
    )

    args, extra_args = parser.parse_known_args()

    # Print banner
    print_header("🌤️  WASHINGTON DC WEATHER DASHBOARD ORCHESTRATOR")

    print(Fore.WHITE + "This script will:")
    print("  1. Check required files and dependencies")
    print("  2. Generate weather forecasts (historical + API + ML)")
    print("  3. Verify outputs were created")
    print("  4. Launch interactive Streamlit dashboard")
    print()

    if args.skip_forecast:
        print_warning("--skip-forecast enabled: Skipping forecast generation")
        print_info("Dashboard will use existing data files")
        print()

    if args.force_retrain:
        print_warning("--force-retrain enabled: ML models will be retrained")
        print()

    # Start workflow
    start_time = time.time()

    # Step 1: Check required files
    if not check_required_files():
        print_error("\n❌ Pre-flight checks failed!")
        print_info("Make sure you're running this script from the project directory:")
        print(Fore.WHITE + f"    cd {SCRIPT_DIR}")
        return 1

    # Step 2: Check dependencies
    if not check_dependencies():
        print_error("\n❌ Dependency checks failed!")
        print_info("Install missing dependencies and try again")
        return 1

    # Step 3: Generate forecasts (unless skipped)
    if not args.skip_forecast:
        success = run_forecast_generation(force_retrain=args.force_retrain)

        if not success:
            print_error("\n❌ Forecast generation failed!")
            response = input(Fore.YELLOW + "\nContinue with existing data? (y/n): ")
            if response.lower() != 'y':
                print_info("Exiting.")
                return 1
            print_info("Continuing with existing data...")
    else:
        print_info("\nSkipping forecast generation (--skip-forecast enabled)")

    # Step 4: Verify outputs
    if not verify_outputs():
        print_warning("\n⚠️  Some outputs are missing!")
        response = input(Fore.YELLOW + "Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print_info("Exiting.")
            return 1

    # Calculate elapsed time
    elapsed = time.time() - start_time
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * 80)
    print(Fore.GREEN + Style.BRIGHT + f"✅ Setup completed in {elapsed:.1f} seconds")
    print(Fore.CYAN + Style.BRIGHT + "=" * 80)
    print()

    # Step 5: Launch Streamlit
    success = launch_streamlit(port=args.port, extra_args=extra_args)

    # Cleanup
    print()
    print_header("👋 Thank you for using Weather Dashboard!", char='-')
    print(Fore.WHITE + "To run again:")
    print(Fore.WHITE + f"    python {Path(__file__).name}")
    print()

    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print_info("\n\n👋 Interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        print()
        print_error(f"\n❌ Unexpected error: {e}")
        print_info("Please report this issue on GitHub")
        sys.exit(1)
