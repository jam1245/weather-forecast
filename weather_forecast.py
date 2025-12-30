import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import logging

# Import ML forecasting module
try:
    from weather_ml_forecast import generate_ml_forecasts, compare_forecasts
    ML_FORECASTING_AVAILABLE = True
except ImportError as e:
    ML_FORECASTING_AVAILABLE = False
    logging.warning(f"ML forecasting not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)

def fetch_historical_data():
    """Fetch 30 days of historical weather data from Open-Meteo Archive API for Washington DC"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)

    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude=38.9072&longitude=-77.0369&"
           f"start_date={start_date}&end_date={end_date}&"
           f"hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America/New_York")

    print("Fetching historical weather data from Open-Meteo Archive API...")
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    hourly_data = data['hourly']

    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly_data['time']),
        'temperature_fahrenheit': hourly_data['temperature_2m'],
        'data_type': 'historical'
    })

    print(f"Historical data: {len(df)} hours from {start_date} to {end_date}")
    return df

def fetch_forecast_data():
    """Fetch 7-day weather forecast from Open-Meteo API for Washington DC"""
    url = "https://api.open-meteo.com/v1/forecast?latitude=38.9072&longitude=-77.0369&hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America/New_York"

    print("Fetching forecast data from Open-Meteo API...")
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    hourly_data = data['hourly']

    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly_data['time']),
        'temperature_fahrenheit': hourly_data['temperature_2m'],
        'data_type': 'forecast'
    })

    print(f"Forecast data: {len(df)} hours")
    return df

def calculate_confidence_intervals(forecast_df):
    """
    Calculate 95% confidence intervals for forecast data.
    Uncertainty increases with forecast horizon (more distant = less certain).
    """
    # Create a copy to avoid modifying original
    df = forecast_df.copy()

    # Calculate hours from now for each forecast point
    now = df['datetime'].min()
    hours_ahead = (df['datetime'] - now).dt.total_seconds() / 3600

    # Uncertainty increases with time: start at ±2°F, increase to ±6°F at 7 days
    # This is a realistic model of forecast uncertainty
    base_uncertainty = 2.0
    max_uncertainty = 6.0
    max_hours = hours_ahead.max()

    uncertainty = base_uncertainty + (max_uncertainty - base_uncertainty) * (hours_ahead / max_hours)

    # 95% confidence interval (approximately ±2 standard deviations)
    df['ci_lower'] = df['temperature_fahrenheit'] - uncertainty
    df['ci_upper'] = df['temperature_fahrenheit'] + uncertainty

    return df

def save_to_csv(historical_df, forecast_df, ml_forecasts=None):
    """
    Save combined temperature data to CSV file with optional ML forecasts

    Args:
        historical_df: Historical temperature data
        forecast_df: API forecast data
        ml_forecasts: Dictionary with ML forecast DataFrames (optional)
    """
    # Start with historical and API forecast data
    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    combined_df = combined_df.sort_values('datetime')

    # Add ML forecast columns if available
    if ml_forecasts and 'prophet' in ml_forecasts:
        print("Adding Prophet ML forecast to CSV...")
        prophet_df = ml_forecasts['prophet'].copy()
        prophet_df = prophet_df.rename(columns={
            'temperature_fahrenheit': 'ml_forecast_prophet',
            'lower_bound': 'ml_forecast_prophet_lower',
            'upper_bound': 'ml_forecast_prophet_upper'
        })

        # Merge on datetime
        combined_df = pd.merge(
            combined_df,
            prophet_df[['datetime', 'ml_forecast_prophet', 'ml_forecast_prophet_lower', 'ml_forecast_prophet_upper']],
            on='datetime',
            how='left'
        )

    if ml_forecasts and 'sarima' in ml_forecasts:
        print("Adding SARIMA ML forecast to CSV...")
        sarima_df = ml_forecasts['sarima'].copy()
        sarima_df = sarima_df.rename(columns={
            'temperature_fahrenheit': 'ml_forecast_sarima',
            'lower_bound': 'ml_forecast_sarima_lower',
            'upper_bound': 'ml_forecast_sarima_upper'
        })

        # Merge on datetime
        combined_df = pd.merge(
            combined_df,
            sarima_df[['datetime', 'ml_forecast_sarima', 'ml_forecast_sarima_lower', 'ml_forecast_sarima_upper']],
            on='datetime',
            how='left'
        )

    csv_filename = 'weather_historical_forecast.csv'
    combined_df.to_csv(csv_filename, index=False)
    print(f"Data saved to {csv_filename}")

    return combined_df

def create_visualization(historical_df, forecast_df, ml_forecasts=None):
    """
    Create a professional visualization showing historical data, API forecast, and ML forecasts

    Args:
        historical_df: Historical temperature data
        forecast_df: API forecast data
        ml_forecasts: Dictionary with ML forecast DataFrames (optional)
    """
    # Calculate confidence intervals for API forecast
    forecast_with_ci = calculate_confidence_intervals(forecast_df)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot historical data (solid blue line)
    ax.plot(historical_df['datetime'], historical_df['temperature_fahrenheit'],
            linewidth=2.5, color='#2E86AB', label='Historical (Actual)', zorder=3)

    # Plot API forecast data (solid red line)
    ax.plot(forecast_with_ci['datetime'], forecast_with_ci['temperature_fahrenheit'],
            linewidth=2.5, color='#E63946', label='API Forecast (Open-Meteo)', zorder=3)

    # Plot 95% confidence interval for API forecast (shaded red area)
    ax.fill_between(forecast_with_ci['datetime'],
                     forecast_with_ci['ci_lower'],
                     forecast_with_ci['ci_upper'],
                     color='#E63946', alpha=0.2, label='API 95% CI', zorder=2)

    # Plot ML forecasts if available
    if ml_forecasts and 'prophet' in ml_forecasts:
        prophet_df = ml_forecasts['prophet']
        ax.plot(prophet_df['datetime'], prophet_df['temperature_fahrenheit'],
                linewidth=2.5, color='#06A77D', label='ML Forecast (Prophet)',
                linestyle='--', zorder=3)

        # Plot Prophet confidence interval (shaded green area)
        ax.fill_between(prophet_df['datetime'],
                         prophet_df['lower_bound'],
                         prophet_df['upper_bound'],
                         color='#06A77D', alpha=0.15, label='Prophet 95% CI', zorder=1)

    if ml_forecasts and 'sarima' in ml_forecasts:
        sarima_df = ml_forecasts['sarima']
        ax.plot(sarima_df['datetime'], sarima_df['temperature_fahrenheit'],
                linewidth=2.0, color='#F77F00', label='ML Forecast (SARIMA)',
                linestyle=':', zorder=3)

    # Add vertical line marking transition from history to forecast
    transition_point = historical_df['datetime'].max()
    ax.axvline(x=transition_point, color='#6C757D', linestyle='--',
               linewidth=2, label='History/Forecast Boundary', zorder=4)

    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temperature (°F)', fontsize=14, fontweight='bold')
    title = 'Washington DC Temperature: Historical + API Forecast'
    if ml_forecasts:
        title += ' + ML Forecast Comparison'
    ax.set_title(title + '\nwith 95% Confidence Intervals',
                fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, shadow=True)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add some statistics as text
    hist_min = historical_df['temperature_fahrenheit'].min()
    hist_max = historical_df['temperature_fahrenheit'].max()
    hist_mean = historical_df['temperature_fahrenheit'].mean()
    forecast_min = forecast_df['temperature_fahrenheit'].min()
    forecast_max = forecast_df['temperature_fahrenheit'].max()

    stats_text = (f"Historical: {hist_min:.1f}°F to {hist_max:.1f}°F (avg: {hist_mean:.1f}°F)\n"
                 f"Forecast: {forecast_min:.1f}°F to {forecast_max:.1f}°F")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    chart_filename = 'temperature_historical_forecast.png'
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {chart_filename}")
    plt.close()

def main():
    """Main function to orchestrate the enhanced weather analysis workflow with ML forecasting"""
    try:
        # Fetch both historical and forecast data
        historical_df = fetch_historical_data()
        forecast_df = fetch_forecast_data()

        # Generate ML forecasts if available
        ml_forecasts = None
        if ML_FORECASTING_AVAILABLE:
            print("\n" + "="*60)
            print("GENERATING ML FORECASTS")
            print("="*60)
            try:
                ml_forecasts = generate_ml_forecasts(
                    historical_df,
                    forecast_periods=168,  # 7 days
                    use_prophet=True,
                    use_sarima=False,  # Disable SARIMA by default (slower)
                    force_retrain=False  # Use cached models if available
                )

                # Compare forecasts if Prophet is available
                if 'prophet' in ml_forecasts:
                    compare_forecasts(
                        forecast_df,
                        ml_forecasts['prophet'],
                        model_name="Prophet"
                    )

            except Exception as e:
                print(f"⚠️  ML forecasting failed: {e}")
                print("Continuing with API forecast only...")
                ml_forecasts = None
        else:
            print("\n⚠️  ML forecasting libraries not installed")
            print("Install with: pip install prophet statsmodels pmdarima scikit-learn")
            print("Continuing with API forecast only...\n")

        # Save combined data to CSV (with ML forecasts if available)
        combined_df = save_to_csv(historical_df, forecast_df, ml_forecasts)

        # Create visualization (with ML forecasts if available)
        create_visualization(historical_df, forecast_df, ml_forecasts)

        # Print summary
        print("\n" + "="*60)
        print("SUCCESS! Weather analysis complete")
        print("="*60)
        print(f"\nHistorical data points: {len(historical_df)}")
        print(f"API Forecast data points: {len(forecast_df)}")

        if ml_forecasts:
            if 'prophet' in ml_forecasts:
                print(f"ML Prophet forecast points: {len(ml_forecasts['prophet'])}")
            if 'sarima' in ml_forecasts:
                print(f"ML SARIMA forecast points: {len(ml_forecasts['sarima'])}")

        print(f"Total data points: {len(combined_df)}")
        print(f"\nHistorical temperature range: {historical_df['temperature_fahrenheit'].min():.1f}°F - {historical_df['temperature_fahrenheit'].max():.1f}°F")
        print(f"API Forecast temperature range: {forecast_df['temperature_fahrenheit'].min():.1f}°F - {forecast_df['temperature_fahrenheit'].max():.1f}°F")

        if ml_forecasts and 'prophet' in ml_forecasts:
            prophet_temps = ml_forecasts['prophet']['temperature_fahrenheit']
            print(f"ML Prophet forecast range: {prophet_temps.min():.1f}°F - {prophet_temps.max():.1f}°F")

        print(f"\nFiles created:")
        print("  - weather_historical_forecast.csv (combined data with ML forecasts)")
        print("  - temperature_historical_forecast.png (visualization)")
        if ml_forecasts:
            print("  - models_cache/ (trained ML models)")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
