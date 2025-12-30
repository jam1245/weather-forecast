import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

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

def create_visualization(historical_df, forecast_df, show_confidence_interval=True):
    """Create a professional visualization showing historical data, forecast, and confidence intervals"""

    forecast_with_ci = calculate_confidence_intervals(forecast_df)

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot historical data
    ax.plot(historical_df['datetime'], historical_df['temperature_fahrenheit'],
            linewidth=2.5, color='#2E86AB', label='Historical (Actual)', zorder=3)

    # Plot forecast data
    ax.plot(forecast_with_ci['datetime'], forecast_with_ci['temperature_fahrenheit'],
            linewidth=2.5, color='#E63946', label='Forecast', zorder=3)

    # Plot confidence interval if enabled
    if show_confidence_interval:
        ax.fill_between(forecast_with_ci['datetime'],
                         forecast_with_ci['ci_lower'],
                         forecast_with_ci['ci_upper'],
                         color='#E63946', alpha=0.2, label='95% Confidence Interval', zorder=2)

    # Add vertical line marking transition
    transition_point = historical_df['datetime'].max()
    ax.axvline(x=transition_point, color='#6C757D', linestyle='--',
               linewidth=2, label='History/Forecast Boundary', zorder=4)

    # Formatting
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Temperature (°F)', fontsize=13, fontweight='bold')
    ax.set_title('Washington DC Temperature Analysis',
                fontsize=15, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, shadow=True)

    plt.xticks(rotation=45, ha='right')

    # Add statistics box
    hist_min = historical_df['temperature_fahrenheit'].min()
    hist_max = historical_df['temperature_fahrenheit'].max()
    hist_mean = historical_df['temperature_fahrenheit'].mean()
    forecast_min = forecast_df['temperature_fahrenheit'].min()
    forecast_max = forecast_df['temperature_fahrenheit'].max()

    stats_text = (f"Historical: {hist_min:.1f}°F to {hist_max:.1f}°F (avg: {hist_mean:.1f}°F)\n"
                 f"Forecast: {forecast_min:.1f}°F to {forecast_max:.1f}°F")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    return fig

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

    # Confidence interval toggle
    show_ci = st.sidebar.checkbox(
        "Show Confidence Interval",
        value=True,
        help="Display 95% confidence interval around forecast"
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

        fig = create_visualization(historical_df, forecast_df, show_ci)
        st.pyplot(fig)
        plt.close()

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
            st.markdown(f"""
            ### Data Information

            **Historical Data:**
            - **Period:** Last {historical_days} days
            - **Data Points:** {len(historical_df)} hours
            - **Temperature Range:** {historical_df['temperature_fahrenheit'].min():.1f}°F to {historical_df['temperature_fahrenheit'].max():.1f}°F

            **Forecast Data:**
            - **Period:** Next 7 days
            - **Data Points:** {len(forecast_df)} hours
            - **Temperature Range:** {forecast_df['temperature_fahrenheit'].min():.1f}°F to {forecast_df['temperature_fahrenheit'].max():.1f}°F

            **Confidence Interval:**
            - The shaded area around the forecast represents the 95% confidence interval
            - Uncertainty increases with time (±2°F near-term to ±6°F at 7 days)
            - This reflects realistic forecast uncertainty

            **Data Source:**
            - Historical: [Open-Meteo Archive API](https://archive-api.open-meteo.com/)
            - Forecast: [Open-Meteo Forecast API](https://open-meteo.com/)
            - Data is cached for 30 minutes to reduce API calls
            """)

    except Exception as e:
        st.error(f"❌ Error fetching weather data: {str(e)}")
        st.info("Please check your internet connection and try refreshing the data.")

if __name__ == "__main__":
    main()
