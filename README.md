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

## 🤖 Machine Learning Forecasting

### AI-Powered Predictions

This project now includes **custom machine learning models** that generate temperature forecasts to compare against professional weather services:

**🟢 Prophet Model (Facebook):**
- Advanced time series forecasting
- Captures daily, weekly, and yearly seasonality patterns
- Provides 95% confidence intervals
- Trains in 30-60 seconds on historical data

**Forecast Comparison Features:**
- **Side-by-side visualization** - Compare API vs ML forecasts
- **Statistical metrics** - Mean difference, correlation, MAE
- **Confidence intervals** - Uncertainty quantification for both forecasts
- **Interactive toggles** - Show/hide each forecast independently

### Why ML Forecasting?

✅ **Educational** - Learn how ML approaches time series prediction
✅ **Comparative Analysis** - See how ML forecasts differ from physics-based models
✅ **Pattern Recognition** - ML excels at identifying historical patterns
✅ **Production-Ready** - Includes caching, error handling, and logging

**Note:** ML forecasts use only historical temperature data, while professional weather services use comprehensive atmospheric models. The comparison demonstrates different forecasting methodologies.

**Learn More:** See [MODEL_INFO.md](MODEL_INFO.md) for detailed model documentation

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

   **Note:** This includes ML forecasting libraries (Prophet, statsmodels). If installation fails, try:
   ```bash
   # Install Prophet separately (can be tricky on some systems)
   conda install -c conda-forge prophet

   # Then install remaining dependencies
   pip install -r requirements.txt
   ```

   **Optional:** Run without ML forecasting (lighter installation):
   ```bash
   pip install streamlit pandas matplotlib requests numpy
   # ML forecasting will be automatically disabled if libraries are missing
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
2. **Show API Forecast**: Toggle the Open-Meteo professional forecast (red line)
3. **Show ML Forecast (Prophet)**: Toggle the machine learning forecast (green dashed line)
4. **Show Confidence Intervals**: Toggle the shaded uncertainty bands around forecasts
5. **Refresh Data**: Click to fetch the latest weather data (clears cache and retrains models)

### Understanding the Visualization
- **Blue Solid Line**: Historical actual temperatures (measured data)
- **Red Solid Line**: API forecast temperatures (Open-Meteo professional service)
- **Green Dashed Line**: ML forecast temperatures (Prophet model)
- **Shaded Red Area**: API forecast 95% confidence interval (if enabled)
  - Near-term forecasts: ±2°F uncertainty
  - 7-day forecasts: ±6°F uncertainty
- **Shaded Green Area**: ML forecast 95% confidence interval (if enabled)
- **Vertical Gray Dashed Line**: Marks where history ends and forecast begins
- **Statistics Box**: Summary of temperature ranges for each dataset

### Forecast Comparison Section
When both API and ML forecasts are enabled, a comparison section shows:
- **Mean Difference**: Average temperature difference between forecasts
- **Correlation**: How well the forecasts agree on trends (0-1 scale)
- **Interpretation**: Which forecast predicts warmer/cooler temperatures
- **Detailed Table**: Hour-by-hour comparison data

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
├── weather_app.py                          # Main Streamlit application (with ML forecasts)
├── weather_forecast.py                     # CLI script for data fetching + ML forecasts
├── weather_ml_forecast.py                  # 🆕 ML forecasting module (Prophet, SARIMA)
├── test_ml_forecast.py                     # 🆕 ML model testing and validation
│
├── requirements.txt                        # Python dependencies (includes ML libraries)
├── README.md                              # This file
├── DATA_INFO.md                           # Detailed data documentation
├── MODEL_INFO.md                          # 🆕 ML model documentation and guide
│
├── START_APP.bat                          # Windows launcher script
├── start_app.sh                           # Unix/Mac launcher script
│
├── Generated files (auto-updated daily):
│   ├── weather_historical_forecast.csv    # Combined data (historical + API + ML forecasts)
│   └── temperature_historical_forecast.png # Visualization with all forecasts
│
└── models_cache/                          # 🆕 Cached trained ML models
    ├── prophet_model.json                 # Serialized Prophet model
    ├── prophet_model.meta                 # Model metadata
    ├── sarima_model.pkl                   # Serialized SARIMA model
    └── sarima_model.meta                  # SARIMA metadata
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
**Core Technologies:**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization
- **Requests**: HTTP library for API calls
- **NumPy**: Numerical computations

**ML Forecasting (New):**
- **Prophet (Facebook)**: Time series forecasting with seasonality
- **Statsmodels**: SARIMA statistical models
- **pmdarima**: Automated ARIMA parameter selection
- **scikit-learn**: Model evaluation metrics

### Performance Optimizations
- `@st.cache_data` decorator with 30-minute TTL (time-to-live)
- Efficient data fetching with consolidated API calls
- Optimized rendering with Streamlit's native components
- **ML Model Caching**: Trained models are cached to avoid retraining (30-60s speedup)
- **Intelligent Retraining**: Models only retrain when historical data changes

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
- **ML Learning**: Understand time series forecasting and model comparison
- **Professional Demos**: Showcase data visualization, web app development, and ML skills
- **Research**: Access historical weather data and compare forecasting methodologies

## 🧪 Testing ML Forecasts

### Quick Test
Test the ML forecasting module with synthetic data:
```bash
python test_ml_forecast.py --mode quick
```

### Standard Test (Recommended)
Run full test suite with 30 days of data:
```bash
python test_ml_forecast.py --mode standard
```

### Comprehensive Test
Test both Prophet and SARIMA models:
```bash
python test_ml_forecast.py --mode comprehensive
```

**What Gets Tested:**
- ✅ Prophet model training and forecasting
- ✅ SARIMA model (comprehensive mode only)
- ✅ Model caching functionality
- ✅ Forecast comparison metrics
- ✅ Edge case handling (missing data, insufficient data, etc.)
- ✅ Output validation (confidence intervals, data types, etc.)

### Manual Testing
Generate forecasts with real data:
```bash
python weather_forecast.py
```

This will:
1. Fetch 30 days of historical data from Open-Meteo
2. Train Prophet model (or load from cache)
3. Generate 7-day ML forecast
4. Compare with API forecast
5. Save combined data to CSV with ML forecast columns

## 🤝 Contributing

This is a demonstration project. Feel free to fork and modify for your own use!

### Potential Enhancements
**Data & Features:**
- Add more locations beyond Washington DC
- Include additional weather parameters (humidity, precipitation, wind speed)
- Add weather alerts and warnings
- Export data to CSV/Excel from the web interface
- Add comparison between different time periods

**ML & Forecasting:**
- Implement additional ML models (LSTM, XGBoost, LightGBM)
- Add ensemble forecasting (combine multiple models)
- Include feature engineering (lagged variables, rolling statistics)
- Implement automated hyperparameter tuning
- Add cross-validation and backtesting
- Create forecast accuracy tracking over time

**Visualization:**
- Add interactive Plotly charts (zoom, pan, hover details)
- Include weather pattern analysis and anomaly detection
- Create forecast divergence plots (where models disagree)
- Add historical forecast accuracy dashboard

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

### ML Forecasting Issues

**Problem:** "Prophet not installed" or "ML forecasting not available"

**Solution:**
```bash
# Try conda (recommended for Prophet)
conda install -c conda-forge prophet

# Or pip
pip install prophet statsmodels pmdarima scikit-learn
```

---

**Problem:** ML forecast training is very slow (>2 minutes)

**Solutions:**
- ✅ First run trains the model (30-60 seconds) - this is normal
- ✅ Subsequent runs use cached models (< 1 second)
- ✅ Clear cache to force retrain: Delete `models_cache/` folder
- ✅ Reduce historical days for faster training

---

**Problem:** ML forecasts look unrealistic or very different from API

**Explanation:**
- This is often expected - ML uses different methodology than physics-based models
- ML only sees historical temperature patterns
- API uses comprehensive atmospheric data and weather models
- Large divergence can indicate:
  - Upcoming weather events (storms, fronts) that ML can't predict
  - Seasonal transitions
  - Unusual recent weather patterns

**Solutions:**
- Check the forecast comparison metrics (correlation, mean difference)
- Low correlation (<0.7) is informative, not necessarily a problem
- For critical decisions, trust the API forecast (professional weather service)

---

**Problem:** "Insufficient data for training" warning

**Solution:**
- Increase historical period to at least 7 days (30 days recommended)
- ML models need sufficient data to learn patterns
- More data = better pattern recognition

## 📞 Support

For issues with the Open-Meteo API, visit: https://open-meteo.com/

---

**Built with ❤️ using Streamlit and Open-Meteo API**

Last Updated: December 2025
