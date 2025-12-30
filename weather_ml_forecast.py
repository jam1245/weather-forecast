"""
Machine Learning Weather Forecasting Module

This module implements time series forecasting models to predict temperature
and compare against professional weather service forecasts (Open-Meteo API).

Models Implemented:
1. Facebook Prophet - Handles seasonality, trends, and special events
2. SARIMA (Seasonal ARIMA) - Statistical baseline for comparison

Author: AI/ML Engineering Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import warnings
import joblib
from pathlib import Path

# ML/Stats libraries
try:
    from prophet import Prophet
    from prophet.serialize import model_to_json, model_from_json
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    import pmdarima as pm
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False
    logging.warning("SARIMA libraries not available. Install with: pip install statsmodels pmdarima")

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings from libraries
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Model cache directory
MODEL_CACHE_DIR = Path("models_cache")
MODEL_CACHE_DIR.mkdir(exist_ok=True)


class WeatherForecaster:
    """Base class for weather forecasting models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.training_data_hash = None

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of training data for cache validation"""
        return str(hash(df.to_json()))

    def _get_cache_path(self) -> Path:
        """Get the cache file path for this model"""
        return MODEL_CACHE_DIR / f"{self.model_name}_model.pkl"

    def save_model(self):
        """Save trained model to cache"""
        raise NotImplementedError

    def load_model(self) -> bool:
        """Load trained model from cache. Returns True if successful."""
        raise NotImplementedError


class ProphetForecaster(WeatherForecaster):
    """Facebook Prophet forecasting implementation"""

    def __init__(self):
        super().__init__("prophet")
        self.forecast_df = None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data in Prophet's required format (ds, y columns)

        Args:
            df: DataFrame with 'datetime' and 'temperature_fahrenheit' columns

        Returns:
            DataFrame with 'ds' (datetime) and 'y' (value) columns
        """
        logger.info(f"Preparing {len(df)} data points for Prophet")

        prophet_df = pd.DataFrame({
            'ds': df['datetime'],
            'y': df['temperature_fahrenheit']
        })

        return prophet_df

    def train(self, df: pd.DataFrame, force_retrain: bool = False) -> None:
        """
        Train Prophet model on historical temperature data

        Args:
            df: Historical temperature DataFrame
            force_retrain: Force retraining even if cached model exists
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Run: pip install prophet")

        data_hash = self._get_data_hash(df)

        # Try to load cached model if data hasn't changed
        if not force_retrain and self.load_model():
            if self.training_data_hash == data_hash:
                logger.info("Using cached Prophet model (data unchanged)")
                return

        logger.info("Training Prophet model...")
        logger.info(f"Training data: {len(df)} hourly observations")

        # Prepare data
        prophet_df = self.prepare_data(df)

        # Configure Prophet model
        # - Hourly data with daily/weekly/yearly seasonality
        # - Growth: linear (temperature trends are relatively linear short-term)
        # - Changepoint prior scale: 0.05 (moderate flexibility)
        self.model = Prophet(
            growth='linear',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            interval_width=0.95,
            uncertainty_samples=1000
        )

        # Add custom seasonalities for better hourly patterns
        self.model.add_seasonality(
            name='hourly',
            period=1,
            fourier_order=8
        )

        # Fit the model
        logger.info("Fitting Prophet model (this may take 30-60 seconds)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)

        self.is_fitted = True
        self.training_data_hash = data_hash

        logger.info("✅ Prophet model trained successfully")

        # Save model to cache
        self.save_model()

    def forecast(self, periods: int = 168, freq: str = 'H') -> pd.DataFrame:
        """
        Generate forecast for specified number of periods

        Args:
            periods: Number of periods to forecast (default: 168 hours = 7 days)
            freq: Frequency of predictions (default: 'H' for hourly)

        Returns:
            DataFrame with columns: datetime, temperature, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before forecasting")

        logger.info(f"Generating {periods}-period forecast with Prophet...")

        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq, include_history=False)

        # Generate forecast
        forecast = self.model.predict(future)

        # Extract relevant columns and rename
        result_df = pd.DataFrame({
            'datetime': forecast['ds'],
            'temperature_fahrenheit': forecast['yhat'],
            'lower_bound': forecast['yhat_lower'],
            'upper_bound': forecast['yhat_upper']
        })

        self.forecast_df = result_df

        logger.info(f"✅ Generated forecast from {result_df['datetime'].min()} to {result_df['datetime'].max()}")

        return result_df

    def save_model(self):
        """Save Prophet model to cache"""
        try:
            cache_path = self._get_cache_path()

            # Prophet's serialization
            with open(cache_path.with_suffix('.json'), 'w') as f:
                f.write(model_to_json(self.model))

            # Save metadata
            metadata = {
                'training_data_hash': self.training_data_hash,
                'is_fitted': self.is_fitted,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(metadata, cache_path.with_suffix('.meta'))

            logger.info(f"Model saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """Load Prophet model from cache"""
        try:
            cache_path = self._get_cache_path()
            json_path = cache_path.with_suffix('.json')
            meta_path = cache_path.with_suffix('.meta')

            if not json_path.exists() or not meta_path.exists():
                return False

            # Load model
            with open(json_path, 'r') as f:
                self.model = model_from_json(f.read())

            # Load metadata
            metadata = joblib.load(meta_path)
            self.training_data_hash = metadata['training_data_hash']
            self.is_fitted = metadata['is_fitted']

            logger.info(f"Loaded cached model from {cache_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}")
            return False


class SARIMAForecaster(WeatherForecaster):
    """SARIMA (Seasonal ARIMA) forecasting implementation"""

    def __init__(self):
        super().__init__("sarima")
        self.best_order = None
        self.best_seasonal_order = None

    def check_stationarity(self, series: pd.Series) -> bool:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test

        Args:
            series: Time series data

        Returns:
            True if series is stationary (p-value < 0.05)
        """
        result = adfuller(series.dropna())
        p_value = result[1]

        is_stationary = p_value < 0.05
        logger.info(f"ADF test p-value: {p_value:.4f} - {'Stationary' if is_stationary else 'Non-stationary'}")

        return is_stationary

    def train(self, df: pd.DataFrame, force_retrain: bool = False) -> None:
        """
        Train SARIMA model using auto_arima for parameter selection

        Args:
            df: Historical temperature DataFrame
            force_retrain: Force retraining even if cached model exists
        """
        if not SARIMA_AVAILABLE:
            raise ImportError("SARIMA libraries not installed. Run: pip install statsmodels pmdarima")

        data_hash = self._get_data_hash(df)

        # Try to load cached model
        if not force_retrain and self.load_model():
            if self.training_data_hash == data_hash:
                logger.info("Using cached SARIMA model (data unchanged)")
                return

        logger.info("Training SARIMA model with auto parameter selection...")
        logger.info(f"Training data: {len(df)} hourly observations")

        # Prepare time series
        ts_data = df.set_index('datetime')['temperature_fahrenheit']

        # Check stationarity
        self.check_stationarity(ts_data)

        # Auto ARIMA to find best parameters
        # For hourly data with daily seasonality: m=24
        # We limit search space for computational efficiency
        logger.info("Running auto_arima (this may take 2-3 minutes)...")

        try:
            self.model = pm.auto_arima(
                ts_data,
                seasonal=True,
                m=24,  # Daily seasonality for hourly data
                max_p=3,
                max_q=3,
                max_P=2,
                max_Q=2,
                max_d=2,
                max_D=1,
                start_p=1,
                start_q=1,
                start_P=1,
                start_Q=1,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                random_state=42,
                n_fits=50
            )

            self.best_order = self.model.order
            self.best_seasonal_order = self.model.seasonal_order
            self.is_fitted = True
            self.training_data_hash = data_hash

            logger.info(f"✅ SARIMA model trained successfully")
            logger.info(f"Best order: {self.best_order}")
            logger.info(f"Best seasonal order: {self.best_seasonal_order}")
            logger.info(f"AIC: {self.model.aic():.2f}")

            # Save model
            self.save_model()

        except Exception as e:
            logger.error(f"SARIMA training failed: {e}")
            raise

    def forecast(self, periods: int = 168) -> pd.DataFrame:
        """
        Generate forecast for specified number of periods

        Args:
            periods: Number of periods to forecast (default: 168 hours = 7 days)

        Returns:
            DataFrame with columns: datetime, temperature, lower_bound, upper_bound
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before forecasting")

        logger.info(f"Generating {periods}-period forecast with SARIMA...")

        # Generate forecast with confidence intervals
        forecast, conf_int = self.model.predict(n_periods=periods, return_conf_int=True, alpha=0.05)

        # Create datetime index for forecast
        # Assumes last training data point + hourly increments
        last_date = datetime.now()
        future_dates = pd.date_range(start=last_date, periods=periods, freq='H')

        # Create result DataFrame
        result_df = pd.DataFrame({
            'datetime': future_dates,
            'temperature_fahrenheit': forecast,
            'lower_bound': conf_int[:, 0],
            'upper_bound': conf_int[:, 1]
        })

        logger.info(f"✅ Generated forecast from {result_df['datetime'].min()} to {result_df['datetime'].max()}")

        return result_df

    def save_model(self):
        """Save SARIMA model to cache"""
        try:
            cache_path = self._get_cache_path()

            # Save model
            joblib.dump(self.model, cache_path)

            # Save metadata
            metadata = {
                'training_data_hash': self.training_data_hash,
                'is_fitted': self.is_fitted,
                'best_order': self.best_order,
                'best_seasonal_order': self.best_seasonal_order,
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(metadata, cache_path.with_suffix('.meta'))

            logger.info(f"Model saved to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """Load SARIMA model from cache"""
        try:
            cache_path = self._get_cache_path()
            meta_path = cache_path.with_suffix('.meta')

            if not cache_path.exists() or not meta_path.exists():
                return False

            # Load model
            self.model = joblib.load(cache_path)

            # Load metadata
            metadata = joblib.load(meta_path)
            self.training_data_hash = metadata['training_data_hash']
            self.is_fitted = metadata['is_fitted']
            self.best_order = metadata.get('best_order')
            self.best_seasonal_order = metadata.get('best_seasonal_order')

            logger.info(f"Loaded cached model from {cache_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load cached model: {e}")
            return False


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics

    Args:
        actual: Actual temperature values
        predicted: Predicted temperature values

    Returns:
        Dictionary with RMSE, MAE, MAPE metrics
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def generate_ml_forecasts(
    historical_df: pd.DataFrame,
    forecast_periods: int = 168,
    use_prophet: bool = True,
    use_sarima: bool = False,
    force_retrain: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Generate ML forecasts using specified models

    Args:
        historical_df: Historical temperature data with columns: datetime, temperature_fahrenheit
        forecast_periods: Number of hours to forecast (default: 168 = 7 days)
        use_prophet: Whether to use Prophet model
        use_sarima: Whether to use SARIMA model
        force_retrain: Force model retraining

    Returns:
        Dictionary with keys 'prophet' and/or 'sarima', values are forecast DataFrames
    """
    logger.info("="*60)
    logger.info("STARTING ML FORECAST GENERATION")
    logger.info("="*60)

    results = {}

    # Validate input data
    if len(historical_df) < 168:  # At least 7 days of data
        logger.warning(f"Insufficient data for training: {len(historical_df)} hours. Need at least 168 hours (7 days)")
        logger.warning("ML forecasts may be unreliable with limited training data")

    # Prophet forecast
    if use_prophet:
        try:
            logger.info("\n📊 PROPHET FORECAST")
            logger.info("-" * 40)

            prophet = ProphetForecaster()
            prophet.train(historical_df, force_retrain=force_retrain)
            forecast_df = prophet.forecast(periods=forecast_periods)

            results['prophet'] = forecast_df

            logger.info(f"Prophet forecast range: {forecast_df['temperature_fahrenheit'].min():.1f}°F to {forecast_df['temperature_fahrenheit'].max():.1f}°F")

        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            logger.exception("Full traceback:")

    # SARIMA forecast
    if use_sarima:
        try:
            logger.info("\n📊 SARIMA FORECAST")
            logger.info("-" * 40)

            sarima = SARIMAForecaster()
            sarima.train(historical_df, force_retrain=force_retrain)
            forecast_df = sarima.forecast(periods=forecast_periods)

            results['sarima'] = forecast_df

            logger.info(f"SARIMA forecast range: {forecast_df['temperature_fahrenheit'].min():.1f}°F to {forecast_df['temperature_fahrenheit'].max():.1f}°F")

        except Exception as e:
            logger.error(f"SARIMA forecast failed: {e}")
            logger.exception("Full traceback:")

    logger.info("\n" + "="*60)
    logger.info(f"ML FORECAST COMPLETE - Generated {len(results)} forecast(s)")
    logger.info("="*60)

    return results


def compare_forecasts(
    api_forecast_df: pd.DataFrame,
    ml_forecast_df: pd.DataFrame,
    model_name: str = "Prophet"
) -> Dict[str, any]:
    """
    Compare ML forecast against API forecast

    Args:
        api_forecast_df: API forecast DataFrame
        ml_forecast_df: ML forecast DataFrame
        model_name: Name of ML model for logging

    Returns:
        Dictionary with comparison statistics
    """
    logger.info(f"\n📊 COMPARING {model_name} vs API FORECAST")
    logger.info("-" * 40)

    # Align forecasts by datetime
    merged = pd.merge(
        api_forecast_df[['datetime', 'temperature_fahrenheit']].rename(columns={'temperature_fahrenheit': 'api_temp'}),
        ml_forecast_df[['datetime', 'temperature_fahrenheit']].rename(columns={'temperature_fahrenheit': 'ml_temp'}),
        on='datetime',
        how='inner'
    )

    if len(merged) == 0:
        logger.warning("No overlapping forecast periods for comparison")
        return {}

    # Calculate differences
    merged['difference'] = merged['ml_temp'] - merged['api_temp']
    merged['abs_difference'] = merged['difference'].abs()

    comparison = {
        'mean_difference': merged['difference'].mean(),
        'mean_abs_difference': merged['abs_difference'].mean(),
        'max_difference': merged['difference'].max(),
        'min_difference': merged['difference'].min(),
        'std_difference': merged['difference'].std(),
        'correlation': merged['api_temp'].corr(merged['ml_temp']),
        'overlap_periods': len(merged)
    }

    logger.info(f"Mean difference: {comparison['mean_difference']:.2f}°F ({model_name} - API)")
    logger.info(f"Mean absolute difference: {comparison['mean_abs_difference']:.2f}°F")
    logger.info(f"Max difference: {comparison['max_difference']:.2f}°F")
    logger.info(f"Correlation: {comparison['correlation']:.3f}")

    # Determine which forecast is more conservative/aggressive
    if comparison['mean_difference'] > 0:
        logger.info(f"→ {model_name} predicts WARMER temperatures on average")
    elif comparison['mean_difference'] < 0:
        logger.info(f"→ {model_name} predicts COOLER temperatures on average")
    else:
        logger.info(f"→ {model_name} and API forecasts are aligned")

    return comparison


if __name__ == "__main__":
    """Test the ML forecasting module"""

    print("\n" + "="*60)
    print("TESTING ML WEATHER FORECASTING MODULE")
    print("="*60 + "\n")

    # Create sample historical data
    print("📥 Creating sample historical data (30 days)...")
    dates = pd.date_range(end=datetime.now(), periods=30*24, freq='H')

    # Simulate temperature with daily and weekly patterns
    hours = np.arange(len(dates))
    base_temp = 50
    daily_pattern = 10 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
    weekly_pattern = 5 * np.sin(2 * np.pi * hours / (24*7))  # Weekly cycle
    noise = np.random.normal(0, 2, len(dates))

    temperatures = base_temp + daily_pattern + weekly_pattern + noise

    sample_df = pd.DataFrame({
        'datetime': dates,
        'temperature_fahrenheit': temperatures
    })

    print(f"✅ Created {len(sample_df)} hourly observations\n")

    # Test forecasting
    forecasts = generate_ml_forecasts(
        sample_df,
        forecast_periods=168,  # 7 days
        use_prophet=True,
        use_sarima=False,  # SARIMA is slower, disable for quick test
        force_retrain=True
    )

    if 'prophet' in forecasts:
        print("\n✅ Prophet forecast successful!")
        print(f"   Forecasted {len(forecasts['prophet'])} periods")
        print(f"   Temperature range: {forecasts['prophet']['temperature_fahrenheit'].min():.1f}°F to {forecasts['prophet']['temperature_fahrenheit'].max():.1f}°F")

    if 'sarima' in forecasts:
        print("\n✅ SARIMA forecast successful!")
        print(f"   Forecasted {len(forecasts['sarima'])} periods")
        print(f"   Temperature range: {forecasts['sarima']['temperature_fahrenheit'].min():.1f}°F to {forecasts['sarima']['temperature_fahrenheit'].max():.1f}°F")

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60 + "\n")
