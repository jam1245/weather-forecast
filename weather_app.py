import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Page configuration
st.set_page_config(
    page_title="Washington DC Weather Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_historical_data(days_back):
    """Fetch historical weather data from Open-Meteo Archive API for Washington DC"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    url = (f"https://archive-api.open-meteo.com/v1/archive?"
           f"latitude=38.9072&longitude=-77.0369&"
           f"start_date={start_date}&end_date={end_date}&"
           f"hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America/New_York")

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    hourly_data = data['hourly']

    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly_data['time']),
        'temperature_fahrenheit': hourly_data['temperature_2m'],
        'data_type': 'historical'
    })

    return df

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_forecast_data():
    """Fetch 7-day weather forecast from Open-Meteo API for Washington DC"""
    url = "https://api.open-meteo.com/v1/forecast?latitude=38.9072&longitude=-77.0369&hourly=temperature_2m&temperature_unit=fahrenheit&timezone=America/New_York"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    hourly_data = data['hourly']

    df = pd.DataFrame({
        'datetime': pd.to_datetime(hourly_data['time']),
        'temperature_fahrenheit': hourly_data['temperature_2m'],
        'data_type': 'forecast'
    })

    return df

def calculate_confidence_intervals(forecast_df):
    """
    Calculate 95% confidence intervals for forecast data.
    Uncertainty increases with forecast horizon (more distant = less certain).
    """
    df = forecast_df.copy()

    now = df['datetime'].min()
    hours_ahead = (df['datetime'] - now).dt.total_seconds() / 3600

    base_uncertainty = 2.0
    max_uncertainty = 6.0
    max_hours = hours_ahead.max()

    uncertainty = base_uncertainty + (max_uncertainty - base_uncertainty) * (hours_ahead / max_hours)

    df['ci_lower'] = df['temperature_fahrenheit'] - uncertainty
    df['ci_upper'] = df['temperature_fahrenheit'] + uncertainty

    return df

def create_visualization(historical_df, forecast_df, show_confidence_interval=True,
                         ml_forecasts=None, show_api=True, show_ml_prophet=True):
    """
    Create an interactive Plotly visualization showing historical data, API forecast, and ML forecasts

    Args:
        historical_df: Historical temperature data
        forecast_df: API forecast data
        show_confidence_interval: Show confidence intervals
        ml_forecasts: Dictionary with ML forecast DataFrames (optional)
        show_api: Show API forecast line
        show_ml_prophet: Show ML Prophet forecast line
    """
    forecast_with_ci = calculate_confidence_intervals(forecast_df)

    # Create figure
    fig = go.Figure()

    # Calculate statistics for annotation
    hist_min = historical_df['temperature_fahrenheit'].min()
    hist_max = historical_df['temperature_fahrenheit'].max()
    hist_mean = historical_df['temperature_fahrenheit'].mean()
    forecast_min = forecast_df['temperature_fahrenheit'].min()
    forecast_max = forecast_df['temperature_fahrenheit'].max()

    # Plot API confidence interval first (if enabled) - so it appears behind the line
    if show_api and show_confidence_interval:
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

    # Plot Prophet confidence interval (if enabled)
    if ml_forecasts and 'prophet' in ml_forecasts and show_ml_prophet and show_confidence_interval:
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

    # Plot API forecast data if enabled
    if show_api:
        fig.add_trace(go.Scatter(
            x=forecast_with_ci['datetime'],
            y=forecast_with_ci['temperature_fahrenheit'],
            mode='lines',
            name='API Forecast (Open-Meteo)',
            line=dict(color='#E63946', width=2.5),
            hovertemplate='<b>API Forecast</b><br>%{y:.1f}°F<br>%{x}<extra></extra>'
        ))

    # Plot ML Prophet forecast if available and enabled
    if ml_forecasts and 'prophet' in ml_forecasts and show_ml_prophet:
        prophet_df = ml_forecasts['prophet']
        fig.add_trace(go.Scatter(
            x=prophet_df['datetime'],
            y=prophet_df['temperature_fahrenheit'],
            mode='lines',
            name='ML Forecast (Prophet)',
            line=dict(color='#06A77D', width=2.5, dash='dash'),
            hovertemplate='<b>ML Forecast (Prophet)</b><br>%{y:.1f}°F<br>%{x}<extra></extra>'
        ))

    # Add vertical line marking transition
    transition_point = historical_df['datetime'].max()
    fig.add_vline(
        x=transition_point,
        line_dash="dash",
        line_color="#6C757D",
        line_width=2,
        annotation_text="History/Forecast Boundary",
        annotation_position="top right"
    )

    # Update layout with professional styling
    fig.update_layout(
        title={
            'text': 'Washington DC Temperature Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2E3033', 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Date',
        yaxis_title='Temperature (°F)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
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
            rangeslider=dict(visible=True),
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

    # Add interactive features
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=14, label="14d", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="rgba(255, 255, 255, 0.9)",
            activecolor="#2E86AB",
            x=0.02,
            y=1.15
        )
    )

    return fig

def compare_forecasts_ui(api_forecast_df, ml_forecast_df):
    """
    Display forecast comparison statistics in the UI

    Args:
        api_forecast_df: API forecast DataFrame
        ml_forecast_df: ML forecast DataFrame
    """
    # Merge forecasts on datetime
    merged = pd.merge(
        api_forecast_df[['datetime', 'temperature_fahrenheit']].rename(columns={'temperature_fahrenheit': 'api_temp'}),
        ml_forecast_df[['datetime', 'temperature_fahrenheit']].rename(columns={'temperature_fahrenheit': 'ml_temp'}),
        on='datetime',
        how='inner'
    )

    if len(merged) == 0:
        st.warning("No overlapping forecast periods for comparison")
        return

    # Calculate differences
    merged['difference'] = merged['ml_temp'] - merged['api_temp']
    merged['abs_difference'] = merged['difference'].abs()

    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        mean_diff = merged['difference'].mean()
        st.metric(
            label="Mean Difference",
            value=f"{mean_diff:.2f}°F",
            delta="ML - API",
            help="Average temperature difference between ML and API forecasts"
        )

    with col2:
        mean_abs_diff = merged['abs_difference'].mean()
        st.metric(
            label="Mean Abs. Difference",
            value=f"{mean_abs_diff:.2f}°F",
            help="Average absolute difference between forecasts"
        )

    with col3:
        correlation = merged['api_temp'].corr(merged['ml_temp'])
        st.metric(
            label="Correlation",
            value=f"{correlation:.3f}",
            help="Pearson correlation between forecasts (1.0 = perfect agreement)"
        )

    with col4:
        max_diff = merged['abs_difference'].max()
        st.metric(
            label="Max Difference",
            value=f"{max_diff:.2f}°F",
            help="Largest temperature difference between forecasts"
        )

    # Interpretation
    st.markdown("#### 📊 Interpretation")

    if mean_diff > 1.0:
        st.info(f"🌡️ **ML forecast predicts WARMER temperatures** on average ({mean_diff:.1f}°F higher than API)")
    elif mean_diff < -1.0:
        st.info(f"❄️ **ML forecast predicts COOLER temperatures** on average ({abs(mean_diff):.1f}°F lower than API)")
    else:
        st.success(f"✅ **Forecasts are well-aligned** (within ±1°F on average)")

    if correlation > 0.9:
        st.success(f"✅ **High correlation** ({correlation:.3f}) - forecasts show similar trends")
    elif correlation > 0.7:
        st.info(f"⚠️ **Moderate correlation** ({correlation:.3f}) - some divergence in trends")
    else:
        st.warning(f"⚠️ **Low correlation** ({correlation:.3f}) - significant differences in forecast patterns")

    # Detailed comparison table
    with st.expander("📋 Detailed Comparison Data"):
        comparison_display = merged.copy()
        comparison_display['datetime'] = comparison_display['datetime'].dt.strftime('%Y-%m-%d %H:%M')
        comparison_display = comparison_display.rename(columns={
            'datetime': 'Date/Time',
            'api_temp': 'API Forecast (°F)',
            'ml_temp': 'ML Forecast (°F)',
            'difference': 'Difference (°F)'
        })

        st.dataframe(
            comparison_display[['Date/Time', 'API Forecast (°F)', 'ML Forecast (°F)', 'Difference (°F)']],
            hide_index=True,
            use_container_width=True,
            height=400
        )


def main():
    # Header
    st.markdown('<div class="main-header">🌤️ Washington DC Weather Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Historical Analysis & 7-Day Forecast with Confidence Intervals</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Historical days selector
    historical_days = st.sidebar.selectbox(
        "Historical Period",
        options=[7, 14, 30, 60],
        index=2,  # Default to 30 days
        help="Select how many days of historical data to display"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Forecast Display")

    # Forecast toggles
    show_api_forecast = st.sidebar.checkbox(
        "Show API Forecast",
        value=True,
        help="Display Open-Meteo API forecast (professional weather service)"
    )

    show_ml_forecast = st.sidebar.checkbox(
        "Show ML Forecast (Prophet)",
        value=ML_FORECASTING_AVAILABLE,
        disabled=not ML_FORECASTING_AVAILABLE,
        help="Display ML-generated forecast using Facebook Prophet"
    )

    # Confidence interval toggle
    show_ci = st.sidebar.checkbox(
        "Show Confidence Intervals",
        value=True,
        help="Display 95% confidence intervals around forecasts"
    )

    st.sidebar.markdown("---")

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Data info
    st.sidebar.markdown("### 📊 Data Source")
    st.sidebar.info("**Open-Meteo API**\n\nHistorical data from archive API\nForecast data updated hourly")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Location:** Washington DC  \n**Coordinates:** 38.9072°N, 77.0369°W")

    # Main content
    try:
        # Fetch data
        with st.spinner('Fetching weather data...'):
            historical_df = fetch_historical_data(historical_days)
            forecast_df = fetch_forecast_data()

        # Generate ML forecasts if enabled
        ml_forecasts = None
        if show_ml_forecast and ML_FORECASTING_AVAILABLE:
            with st.spinner('Generating ML forecasts (this may take 30-60 seconds)...'):
                try:
                    ml_forecasts = generate_ml_forecasts(
                        historical_df,
                        forecast_periods=168,  # 7 days
                        use_prophet=True,
                        use_sarima=False,
                        force_retrain=False
                    )
                except Exception as e:
                    st.warning(f"⚠️ ML forecasting failed: {str(e)}")
                    st.info("Continuing with API forecast only...")

        # Get current time and latest temperature
        current_time = datetime.now()
        latest_temp = historical_df['temperature_fahrenheit'].iloc[-1]

        # Calculate statistics
        forecast_high = forecast_df['temperature_fahrenheit'].max()
        forecast_low = forecast_df['temperature_fahrenheit'].min()
        hist_avg = historical_df['temperature_fahrenheit'].mean()

        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="🌡️ Current Temperature",
                value=f"{latest_temp:.1f}°F",
                delta=f"{latest_temp - hist_avg:.1f}°F from avg"
            )

        with col2:
            st.metric(
                label="📈 Forecast High",
                value=f"{forecast_high:.1f}°F",
                delta=None
            )

        with col3:
            st.metric(
                label="📉 Forecast Low",
                value=f"{forecast_low:.1f}°F",
                delta=None
            )

        with col4:
            st.metric(
                label="📊 Historical Average",
                value=f"{hist_avg:.1f}°F",
                delta=None
            )

        with col5:
            st.metric(
                label="📅 Data Points",
                value=f"{len(historical_df) + len(forecast_df)}",
                delta=None
            )

        # Last updated timestamp
        st.caption(f"⏰ Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} EST")

        st.markdown("---")

        # Visualization
        st.subheader("📈 Temperature Trend Analysis")

        fig = create_visualization(
            historical_df,
            forecast_df,
            show_confidence_interval=show_ci,
            ml_forecasts=ml_forecasts,
            show_api=show_api_forecast,
            show_ml_prophet=show_ml_forecast
        )
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'weather_forecast',
                'height': 800,
                'width': 1400,
                'scale': 2
            }
        })

        # Forecast Comparison Section
        if ml_forecasts and 'prophet' in ml_forecasts and show_api_forecast and show_ml_forecast:
            st.markdown("---")
            st.subheader("🔄 Forecast Comparison: API vs ML")

            # Compare forecasts
            comparison_stats = compare_forecasts_ui(forecast_df, ml_forecasts['prophet'])

        st.markdown("---")

        # Forecast table
        st.subheader("📋 7-Day Forecast Details")

        # Prepare forecast data for display
        forecast_display = forecast_df.copy()
        forecast_display['Date'] = forecast_display['datetime'].dt.strftime('%Y-%m-%d')
        forecast_display['Time'] = forecast_display['datetime'].dt.strftime('%H:%M')
        forecast_display['Temperature (°F)'] = forecast_display['temperature_fahrenheit'].round(1)

        # Group by date and get daily stats
        daily_forecast = forecast_display.groupby('Date').agg({
            'Temperature (°F)': ['min', 'max', 'mean']
        }).round(1)

        daily_forecast.columns = ['Low (°F)', 'High (°F)', 'Average (°F)']
        daily_forecast = daily_forecast.reset_index()

        # Display tables in columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Daily Summary**")
            st.dataframe(
                daily_forecast,
                hide_index=True,
                use_container_width=True
            )

        with col2:
            st.markdown("**Next 24 Hours (Hourly)**")
            next_24h = forecast_display[['Date', 'Time', 'Temperature (°F)']].head(24)
            st.dataframe(
                next_24h,
                hide_index=True,
                use_container_width=True,
                height=400
            )

        # Data summary
        st.markdown("---")
        with st.expander("ℹ️ About This Dashboard"):
            dashboard_info = f"""
            ### Data Information

            **Historical Data:**
            - **Period:** Last {historical_days} days
            - **Data Points:** {len(historical_df)} hours
            - **Temperature Range:** {historical_df['temperature_fahrenheit'].min():.1f}°F to {historical_df['temperature_fahrenheit'].max():.1f}°F

            **API Forecast Data (Open-Meteo):**
            - **Period:** Next 7 days
            - **Data Points:** {len(forecast_df)} hours
            - **Temperature Range:** {forecast_df['temperature_fahrenheit'].min():.1f}°F to {forecast_df['temperature_fahrenheit'].max():.1f}°F
            - **Source:** Professional weather service with numerical weather models
            """

            if ml_forecasts and 'prophet' in ml_forecasts:
                prophet_temps = ml_forecasts['prophet']['temperature_fahrenheit']
                dashboard_info += f"""

            **ML Forecast Data (Prophet):**
            - **Model:** Facebook Prophet (time series forecasting)
            - **Training Data:** {len(historical_df)} hours of historical temperatures
            - **Forecast Points:** {len(ml_forecasts['prophet'])} hours
            - **Temperature Range:** {prophet_temps.min():.1f}°F to {prophet_temps.max():.1f}°F
            - **Features:** Captures daily/weekly/yearly seasonality patterns
                """

            dashboard_info += """

            **Confidence Intervals:**
            - The shaded areas around forecasts represent 95% confidence intervals
            - API forecast: Uncertainty increases from ±2°F (near-term) to ±6°F (7 days)
            - ML forecast: Model-generated uncertainty based on historical variance
            - This reflects realistic forecast uncertainty

            **Data Sources:**
            - Historical: [Open-Meteo Archive API](https://archive-api.open-meteo.com/)
            - API Forecast: [Open-Meteo Forecast API](https://open-meteo.com/)
            - ML Forecast: Trained on historical data using Prophet algorithm
            - Data is cached for 30 minutes to reduce API calls
            """

            st.markdown(dashboard_info)

    except Exception as e:
        st.error(f"❌ Error fetching weather data: {str(e)}")
        st.info("Please check your internet connection and try refreshing the data.")

if __name__ == "__main__":
    main()
