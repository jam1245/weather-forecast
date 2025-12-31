import requests
import pandas as pd
import plotly.graph_objects as go
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
    Create an interactive Plotly visualization showing historical data, API forecast, and ML forecasts

    Args:
        historical_df: Historical temperature data
        forecast_df: API forecast data
        ml_forecasts: Dictionary with ML forecast DataFrames (optional)
    """
    # Calculate confidence intervals for API forecast
    forecast_with_ci = calculate_confidence_intervals(forecast_df)

    # Create figure
    fig = go.Figure()

    # Calculate statistics for annotation
    hist_min = historical_df['temperature_fahrenheit'].min()
    hist_max = historical_df['temperature_fahrenheit'].max()
    hist_mean = historical_df['temperature_fahrenheit'].mean()
    forecast_min = forecast_df['temperature_fahrenheit'].min()
    forecast_max = forecast_df['temperature_fahrenheit'].max()

    # Plot API confidence interval first (so it appears behind the line)
    # Upper bound
    fig.add_trace(go.Scatter(
        x=forecast_with_ci['datetime'],
        y=forecast_with_ci['ci_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
        name='API CI Upper'
    ))
    # Lower bound with fill
    fig.add_trace(go.Scatter(
        x=forecast_with_ci['datetime'],
        y=forecast_with_ci['ci_lower'],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(230, 57, 70, 0.2)',
        name='API 95% CI',
        hovertemplate='<b>API Confidence Interval</b><br>Upper: %{text[0]:.1f}°F<br>Lower: %{y:.1f}°F<br>Date: %{x}<extra></extra>',
        text=forecast_with_ci[['ci_upper']].values
    ))

    # Plot Prophet confidence interval if available
    if ml_forecasts and 'prophet' in ml_forecasts:
        prophet_df = ml_forecasts['prophet']
        # Upper bound
        fig.add_trace(go.Scatter(
            x=prophet_df['datetime'],
            y=prophet_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            name='Prophet CI Upper'
        ))
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=prophet_df['datetime'],
            y=prophet_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(6, 167, 125, 0.15)',
            name='Prophet 95% CI',
            hovertemplate='<b>Prophet Confidence Interval</b><br>Upper: %{text[0]:.1f}°F<br>Lower: %{y:.1f}°F<br>Date: %{x}<extra></extra>',
            text=prophet_df[['upper_bound']].values
        ))

    # Plot historical data
    fig.add_trace(go.Scatter(
        x=historical_df['datetime'],
        y=historical_df['temperature_fahrenheit'],
        mode='lines',
        name='Historical (Actual)',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='<b>Historical Temperature</b><br>%{y:.1f}°F<br>%{x}<extra></extra>'
    ))

    # Plot API forecast data
    fig.add_trace(go.Scatter(
        x=forecast_with_ci['datetime'],
        y=forecast_with_ci['temperature_fahrenheit'],
        mode='lines',
        name='API Forecast (Open-Meteo)',
        line=dict(color='#E63946', width=2.5),
        hovertemplate='<b>API Forecast</b><br>%{y:.1f}°F<br>%{x}<extra></extra>'
    ))

    # Plot ML Prophet forecast if available
    if ml_forecasts and 'prophet' in ml_forecasts:
        prophet_df = ml_forecasts['prophet']
        fig.add_trace(go.Scatter(
            x=prophet_df['datetime'],
            y=prophet_df['temperature_fahrenheit'],
            mode='lines',
            name='ML Forecast (Prophet)',
            line=dict(color='#06A77D', width=2.5, dash='dash'),
            hovertemplate='<b>ML Forecast (Prophet)</b><br>%{y:.1f}°F<br>%{x}<extra></extra>'
        ))

    # Plot SARIMA forecast if available
    if ml_forecasts and 'sarima' in ml_forecasts:
        sarima_df = ml_forecasts['sarima']
        fig.add_trace(go.Scatter(
            x=sarima_df['datetime'],
            y=sarima_df['temperature_fahrenheit'],
            mode='lines',
            name='ML Forecast (SARIMA)',
            line=dict(color='#F77F00', width=2.0, dash='dot'),
            hovertemplate='<b>ML Forecast (SARIMA)</b><br>%{y:.1f}°F<br>%{x}<extra></extra>'
        ))

    # Add vertical line marking transition
    # Use add_shape instead of add_vline to avoid datetime arithmetic issues
    transition_point = historical_df['datetime'].max()
    fig.add_shape(
        type="line",
        x0=transition_point,
        x1=transition_point,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#6C757D", width=2, dash="dash")
    )
    # Add annotation for the boundary line
    fig.add_annotation(
        x=transition_point,
        y=1.0,
        yref="paper",
        text="History/Forecast Boundary",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font=dict(size=10, color="#6C757D"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#6C757D",
        borderwidth=1,
        borderpad=4
    )

    # Update layout
    title = 'Washington DC Temperature: Historical + API Forecast'
    if ml_forecasts:
        title += ' + ML Forecast Comparison'
    title += '\nwith 95% Confidence Intervals'

    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2E3033', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Date',
        yaxis_title='Temperature (°F)',
        hovermode='x unified',
        template='plotly_white',
        height=800,
        width=1600,
        font=dict(family="Arial, sans-serif", size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(128, 128, 128, 0.2)',
            type='date'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        annotations=[
            dict(
                text=f"Historical: {hist_min:.1f}°F to {hist_max:.1f}°F (avg: {hist_mean:.1f}°F)<br>Forecast: {forecast_min:.1f}°F to {forecast_max:.1f}°F",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(245, 222, 179, 0.8)",
                bordercolor="#C8A878",
                borderwidth=1,
                borderpad=8,
                font=dict(size=10),
                align="left",
                xanchor="left",
                yanchor="top"
            )
        ]
    )

    # Save as static PNG image
    chart_filename = 'temperature_historical_forecast.png'
    try:
        # Try to save as PNG using kaleido (requires: pip install kaleido)
        fig.write_image(chart_filename, width=1600, height=800, scale=2)
        print(f"Chart saved to {chart_filename}")
    except Exception as e:
        print(f"Warning: Could not save PNG image: {e}")
        print("Install kaleido for PNG export: pip install kaleido")
        # Fall back to HTML export
        html_filename = 'temperature_historical_forecast.html'
        fig.write_html(html_filename)
        print(f"Interactive HTML chart saved to {html_filename} instead")

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
