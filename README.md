# 🌤️ Washington DC Weather Dashboard

A professional, interactive web application for analyzing historical weather data and forecasting future temperatures for Washington DC. Built with Streamlit and powered by the Open-Meteo API.

![Dashboard Preview](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Daily Weather Update](https://github.com/jam1245/weather-forecast/actions/workflows/daily-weather-update.yml/badge.svg)

## ✨ Features

### 📊 Interactive Dashboard
- **Real-time Weather Data**: Fetches live historical and forecast data from Open-Meteo API
- **Customizable Historical Period**: Choose from 7, 14, 30, or 60 days of historical data
- **Confidence Intervals**: Toggle 95% confidence intervals around forecasts (uncertainty increases with time)
- **Professional Visualizations**: High-quality matplotlib charts with clear boundaries between historical and forecast data

### 📈 Key Metrics
- Current temperature with deviation from historical average
- 7-day forecast high and low temperatures
- Historical average temperature
- Total data points analyzed
- Last updated timestamp

### 📋 Data Tables
- **Daily Summary**: Min, max, and average temperatures for each day in the forecast
- **Hourly Forecast**: Next 24 hours of temperature predictions
- **Detailed Statistics**: Expandable section with comprehensive data information

### 🎨 Professional Design
- Clean, modern interface with custom styling
- Responsive layout that works on different screen sizes
- Color-coded metrics and visualizations
- Easy-to-use sidebar controls
- Data caching for optimal performance (30-minute cache)

## 🔄 Data Freshness & Automation

### Automated Daily Updates

This repository includes a **GitHub Actions workflow** that automatically fetches and updates weather data every day:

- **Schedule**: Runs daily at 6:00 AM UTC (1:00 AM EST)
- **What Gets Updated**:
  - `weather_historical_forecast.csv` - Latest 30 days of historical data + 7-day forecast
  - `temperature_historical_forecast.png` - Updated visualization
  - `DATA_INFO.md` - Timestamp of last update

### Manual Workflow Triggering

You can manually trigger a fresh data update at any time:

1. Navigate to the [Actions tab](https://github.com/jam1245/weather-forecast/actions)
2. Click on "Daily Weather Data Update" workflow
3. Click the "Run workflow" button (top right)
4. Select the `main` branch
5. Click "Run workflow" to start

The workflow will:
1. ✅ Fetch fresh weather data from Open-Meteo API
2. ✅ Generate updated CSV and PNG files
3. ✅ Automatically commit and push the changes
4. ✅ Update the last-updated timestamp

### Data Information

For detailed information about the data structure, update process, and retention policy, see [DATA_INFO.md](DATA_INFO.md).

**Current Data Includes**:
- 30 days of historical hourly temperature data
- 7 days (168 hours) of forecast data
- Hourly resolution for all data points
- Location: Washington DC (38.9072°N, 77.0369°W)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this project** to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd jam_data_test
   ```

3. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or if you're using Anaconda/Miniconda:
   ```bash
   conda install streamlit pandas matplotlib requests numpy
   ```

### Running the Application

Simply run the following command in your terminal:

```bash
streamlit run weather_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📱 How to Use

### Main Dashboard
- The dashboard loads automatically with 30 days of historical data and a 7-day forecast
- View current temperature, forecast highs/lows, and key statistics at the top
- Scroll down to see the interactive chart and detailed forecast tables

### Sidebar Controls
1. **Historical Period**: Select how many days of past data to display (7, 14, 30, or 60 days)
2. **Show Confidence Interval**: Toggle the shaded uncertainty band around forecasts
3. **Refresh Data**: Click to fetch the latest weather data (clears cache)

### Understanding the Visualization
- **Blue Line**: Historical actual temperatures (measured data)
- **Red Line**: Forecast temperatures (predicted data)
- **Shaded Red Area**: 95% confidence interval (if enabled)
  - Near-term forecasts: ±2°F uncertainty
  - 7-day forecasts: ±6°F uncertainty
- **Vertical Dashed Line**: Marks where history ends and forecast begins
- **Statistics Box**: Summary of temperature ranges

### Data Tables
- **Daily Summary Table**: Shows min, max, and average temperatures for each day
- **Next 24 Hours Table**: Hourly breakdown of temperature predictions

## 🔧 Project Structure

```
weather-forecast/
│
├── .github/
│   └── workflows/
│       └── daily-weather-update.yml       # GitHub Actions automation workflow
│
├── weather_app.py                          # Main Streamlit application
├── weather_forecast.py                     # CLI script for data fetching
├── requirements.txt                        # Python dependencies
├── README.md                              # This file
├── DATA_INFO.md                           # Detailed data documentation
│
├── START_APP.bat                          # Windows launcher script
├── start_app.sh                           # Unix/Mac launcher script
│
└── Generated files (auto-updated daily):
    ├── weather_historical_forecast.csv    # Combined historical + forecast data
    └── temperature_historical_forecast.png # Visualization image
```

## 📊 Data Sources

This application uses the **Open-Meteo API** which provides:

- **Historical Data**: [Archive API](https://archive-api.open-meteo.com/)
  - Actual measured temperatures from weather stations
  - Available for past dates with hourly resolution

- **Forecast Data**: [Forecast API](https://open-meteo.com/)
  - 7-day hourly temperature predictions
  - Updated regularly throughout the day

**Location**: Washington DC (38.9072°N, 77.0369°W)

**Note**: Data is cached for 30 minutes to reduce API calls and improve performance. Click "Refresh Data" to force a fresh fetch.

## 🛠️ Technical Details

### Technologies Used
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Requests**: HTTP library for API calls
- **NumPy**: Numerical computations

### Performance Optimizations
- `@st.cache_data` decorator with 30-minute TTL (time-to-live)
- Efficient data fetching with consolidated API calls
- Optimized rendering with Streamlit's native components

### Confidence Interval Methodology
The 95% confidence interval is calculated using a realistic uncertainty model:
- **Base uncertainty**: ±2°F for near-term forecasts
- **Maximum uncertainty**: ±6°F for 7-day forecasts
- **Linear increase**: Uncertainty grows linearly with forecast horizon
- This reflects how forecast accuracy decreases over time

## 💡 Use Cases

- **Personal Weather Planning**: Check detailed forecasts before planning outdoor activities
- **Data Analysis**: Study temperature trends and patterns in Washington DC
- **Educational**: Learn about forecast uncertainty and confidence intervals
- **Professional Demos**: Showcase data visualization and web app development skills
- **Research**: Access historical weather data for analysis

## 🤝 Contributing

This is a demonstration project. Feel free to fork and modify for your own use!

### Potential Enhancements
- Add more locations beyond Washington DC
- Include additional weather parameters (humidity, precipitation, wind speed)
- Add weather alerts and warnings
- Export data to CSV/Excel from the web interface
- Add comparison between different time periods
- Include weather pattern analysis and trends

## 📄 License

This project uses the Open-Meteo API which is free for non-commercial use.

## 🙋 Troubleshooting

### Port Already in Use
If you see an error about port 8501 being in use:
```bash
streamlit run weather_app.py --server.port 8502
```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

### API Connection Issues
- Check your internet connection
- The Open-Meteo API might be temporarily unavailable
- Try clicking "Refresh Data" after a few moments

### Slow Performance
- The first load might take a few seconds to fetch data
- Subsequent loads use cached data for faster performance
- Clear cache using the "Refresh Data" button if needed

## 📞 Support

For issues with the Open-Meteo API, visit: https://open-meteo.com/

---

**Built with ❤️ using Streamlit and Open-Meteo API**

Last Updated: December 2025
