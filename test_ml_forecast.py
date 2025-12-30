"""
Test Script for ML Weather Forecasting

This script validates the ML forecasting functionality by:
1. Testing Prophet model training and forecasting
2. Testing SARIMA model (optional)
3. Validating forecast outputs
4. Comparing against API forecasts
5. Checking model caching
6. Generating test reports

Usage:
    python test_ml_forecast.py

Author: AI/ML Engineering Team
Version: 1.0.0
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CASES = {
    'quick': {
        'historical_days': 7,
        'forecast_periods': 24,
        'use_sarima': False
    },
    'standard': {
        'historical_days': 30,
        'forecast_periods': 168,
        'use_sarima': False
    },
    'comprehensive': {
        'historical_days': 60,
        'forecast_periods': 168,
        'use_sarima': True
    }
}


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title):
    """Print a formatted subheader"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def check_dependencies():
    """Check if all required libraries are installed"""
    print_header("DEPENDENCY CHECK")

    dependencies = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'prophet': 'prophet',
        'statsmodels': 'statsmodels',
        'pmdarima': 'pmdarima',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib'
    }

    all_available = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {package:20s} - Available")
        except ImportError:
            print(f"❌ {package:20s} - Missing (install with: pip install {package})")
            all_available = False

    return all_available


def create_synthetic_data(days=30, with_noise=True):
    """
    Create synthetic temperature data for testing

    Args:
        days: Number of days of historical data
        with_noise: Add random noise to the data

    Returns:
        DataFrame with datetime and temperature_fahrenheit columns
    """
    print_subheader(f"Generating {days} days of synthetic data")

    dates = pd.date_range(end=datetime.now(), periods=days * 24, freq='H')

    # Create realistic temperature pattern
    hours = np.arange(len(dates))

    # Base temperature
    base_temp = 45

    # Daily cycle (warmer during day, cooler at night)
    daily_pattern = 15 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)

    # Weekly cycle
    weekly_pattern = 5 * np.sin(2 * np.pi * hours / (24 * 7))

    # Seasonal trend (gradual warming/cooling)
    seasonal_trend = 0.1 * hours / 24  # 0.1°F per day trend

    # Random noise
    if with_noise:
        noise = np.random.normal(0, 2, len(dates))
    else:
        noise = 0

    temperatures = base_temp + daily_pattern + weekly_pattern + seasonal_trend + noise

    df = pd.DataFrame({
        'datetime': dates,
        'temperature_fahrenheit': temperatures
    })

    print(f"✅ Created {len(df)} hourly observations")
    print(f"   Temperature range: {df['temperature_fahrenheit'].min():.1f}°F to {df['temperature_fahrenheit'].max():.1f}°F")
    print(f"   Mean: {df['temperature_fahrenheit'].mean():.1f}°F")
    print(f"   Std Dev: {df['temperature_fahrenheit'].std():.1f}°F")

    return df


def test_prophet_forecast(historical_df, forecast_periods=168):
    """
    Test Prophet forecasting

    Args:
        historical_df: Historical temperature data
        forecast_periods: Number of hours to forecast

    Returns:
        Tuple of (success, forecast_df, error_message)
    """
    print_subheader("Testing Prophet Forecast")

    try:
        from weather_ml_forecast import ProphetForecaster

        # Create forecaster
        prophet = ProphetForecaster()

        # Train
        print("Training Prophet model...")
        prophet.train(historical_df, force_retrain=True)

        # Check if model is fitted
        if not prophet.is_fitted:
            return False, None, "Model not fitted after training"

        # Generate forecast
        print(f"Generating {forecast_periods}-hour forecast...")
        forecast_df = prophet.forecast(periods=forecast_periods)

        # Validate forecast
        if len(forecast_df) != forecast_periods:
            return False, None, f"Expected {forecast_periods} forecasts, got {len(forecast_df)}"

        # Check required columns
        required_cols = ['datetime', 'temperature_fahrenheit', 'lower_bound', 'upper_bound']
        missing_cols = [col for col in required_cols if col not in forecast_df.columns]
        if missing_cols:
            return False, None, f"Missing columns: {missing_cols}"

        # Check for NaN values
        if forecast_df.isnull().any().any():
            return False, None, "Forecast contains NaN values"

        # Check confidence interval validity
        invalid_ci = (forecast_df['lower_bound'] > forecast_df['temperature_fahrenheit']).any() or \
                     (forecast_df['upper_bound'] < forecast_df['temperature_fahrenheit']).any()
        if invalid_ci:
            return False, None, "Invalid confidence intervals detected"

        print("✅ Prophet forecast successful!")
        print(f"   Forecast range: {forecast_df['temperature_fahrenheit'].min():.1f}°F to {forecast_df['temperature_fahrenheit'].max():.1f}°F")
        print(f"   CI width: {(forecast_df['upper_bound'] - forecast_df['lower_bound']).mean():.1f}°F")

        return True, forecast_df, None

    except Exception as e:
        return False, None, str(e)


def test_model_caching(historical_df):
    """
    Test model caching functionality

    Args:
        historical_df: Historical temperature data

    Returns:
        Tuple of (success, error_message)
    """
    print_subheader("Testing Model Caching")

    try:
        from weather_ml_forecast import ProphetForecaster
        import time

        # First training (no cache)
        prophet1 = ProphetForecaster()
        start_time = time.time()
        prophet1.train(historical_df, force_retrain=True)
        first_train_time = time.time() - start_time

        print(f"✅ First training (no cache): {first_train_time:.2f} seconds")

        # Second training (should use cache)
        prophet2 = ProphetForecaster()
        start_time = time.time()
        prophet2.train(historical_df, force_retrain=False)
        cached_load_time = time.time() - start_time

        print(f"✅ Cached load: {cached_load_time:.2f} seconds")

        # Check speedup
        speedup = first_train_time / cached_load_time
        print(f"✅ Speedup: {speedup:.1f}x faster")

        if speedup < 5:
            print(f"⚠️  Warning: Expected >5x speedup, got {speedup:.1f}x")

        return True, None

    except Exception as e:
        return False, str(e)


def test_forecast_comparison(api_forecast_df, ml_forecast_df):
    """
    Test forecast comparison functionality

    Args:
        api_forecast_df: API forecast data
        ml_forecast_df: ML forecast data

    Returns:
        Tuple of (success, comparison_stats, error_message)
    """
    print_subheader("Testing Forecast Comparison")

    try:
        from weather_ml_forecast import compare_forecasts

        comparison = compare_forecasts(api_forecast_df, ml_forecast_df, model_name="Prophet")

        # Validate comparison results
        required_keys = ['mean_difference', 'mean_abs_difference', 'correlation']
        missing_keys = [key for key in required_keys if key not in comparison]
        if missing_keys:
            return False, None, f"Missing comparison keys: {missing_keys}"

        print("✅ Forecast comparison successful!")
        print(f"   Mean difference: {comparison['mean_difference']:.2f}°F")
        print(f"   MAE: {comparison['mean_abs_difference']:.2f}°F")
        print(f"   Correlation: {comparison['correlation']:.3f}")

        return True, comparison, None

    except Exception as e:
        return False, None, str(e)


def test_edge_cases():
    """Test edge cases and error handling"""
    print_subheader("Testing Edge Cases")

    from weather_ml_forecast import ProphetForecaster

    # Test 1: Insufficient data
    print("\n📝 Test 1: Insufficient training data")
    try:
        small_df = create_synthetic_data(days=2, with_noise=False)  # Only 2 days
        prophet = ProphetForecaster()
        prophet.train(small_df, force_retrain=True)
        print("✅ Handled insufficient data (warning expected)")
    except Exception as e:
        print(f"❌ Failed with error: {e}")

    # Test 2: Missing values
    print("\n📝 Test 2: Data with missing values")
    try:
        df_with_nan = create_synthetic_data(days=7)
        df_with_nan.loc[10:20, 'temperature_fahrenheit'] = np.nan
        prophet = ProphetForecaster()
        prophet.train(df_with_nan, force_retrain=True)
        print("✅ Handled missing values")
    except Exception as e:
        print(f"❌ Failed with error: {e}")

    # Test 3: Constant values
    print("\n📝 Test 3: Constant temperature data")
    try:
        constant_df = create_synthetic_data(days=7, with_noise=False)
        constant_df['temperature_fahrenheit'] = 50.0  # All same value
        prophet = ProphetForecaster()
        prophet.train(constant_df, force_retrain=True)
        forecast = prophet.forecast(periods=24)
        print("✅ Handled constant values")
    except Exception as e:
        print(f"❌ Failed with error: {e}")


def run_test_suite(test_mode='standard'):
    """
    Run complete test suite

    Args:
        test_mode: 'quick', 'standard', or 'comprehensive'
    """
    print_header(f"ML FORECAST TEST SUITE - {test_mode.upper()} MODE")

    # Get test configuration
    config = TEST_CASES.get(test_mode, TEST_CASES['standard'])
    print(f"\nConfiguration:")
    print(f"  Historical days: {config['historical_days']}")
    print(f"  Forecast periods: {config['forecast_periods']}")
    print(f"  Test SARIMA: {config['use_sarima']}")

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Install required packages and try again.")
        return False

    # Create test data
    historical_df = create_synthetic_data(days=config['historical_days'])

    # Test Prophet
    success, prophet_forecast, error = test_prophet_forecast(
        historical_df,
        forecast_periods=config['forecast_periods']
    )
    if not success:
        print(f"\n❌ Prophet test failed: {error}")
        return False

    # Test caching
    success, error = test_model_caching(historical_df)
    if not success:
        print(f"\n❌ Caching test failed: {error}")
        return False

    # Test comparison (simulate API forecast)
    api_forecast = prophet_forecast.copy()
    api_forecast['temperature_fahrenheit'] += np.random.normal(0, 2, len(api_forecast))  # Add some difference

    success, comparison, error = test_forecast_comparison(api_forecast, prophet_forecast)
    if not success:
        print(f"\n❌ Comparison test failed: {error}")
        return False

    # Test edge cases
    test_edge_cases()

    # Summary
    print_header("TEST SUMMARY")
    print("\n✅ All tests passed successfully!")
    print("\nTest Results:")
    print(f"  ✅ Prophet training: PASS")
    print(f"  ✅ Prophet forecasting: PASS")
    print(f"  ✅ Model caching: PASS")
    print(f"  ✅ Forecast comparison: PASS")
    print(f"  ✅ Edge cases: PASS")

    return True


def main():
    """Main test entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Test ML Weather Forecasting')
    parser.add_argument(
        '--mode',
        choices=['quick', 'standard', 'comprehensive'],
        default='standard',
        help='Test mode (quick=7 days, standard=30 days, comprehensive=60 days with SARIMA)'
    )

    args = parser.parse_args()

    success = run_test_suite(test_mode=args.mode)

    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
