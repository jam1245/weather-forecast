# 🤖 Machine Learning Forecast Models

## Overview

This document explains the machine learning models used for temperature forecasting in the Washington DC Weather Dashboard. These ML forecasts are compared against professional weather service forecasts (Open-Meteo API) to demonstrate different forecasting approaches.

---

## 📊 Models Implemented

### 1. Facebook Prophet

**Type:** Additive time series model
**Status:** ✅ Implemented (Primary ML model)
**Library:** `prophet >= 1.1.5`

#### What is Prophet?

Prophet is an open-source forecasting library developed by Facebook (Meta) designed for business forecasting problems with:
- **Strong seasonal patterns** (daily, weekly, yearly)
- **Missing data tolerance**
- **Trend changes** (changepoints)
- **Holiday effects**

#### Why Prophet for Weather Forecasting?

✅ **Handles Multiple Seasonalities:**
- Hourly patterns (temperature cycles within a day)
- Daily patterns (day/night cycles)
- Weekly patterns (weekday/weekend variations)
- Yearly patterns (seasonal changes)

✅ **Robust to Missing Data:**
- Can handle gaps in historical data
- Doesn't require perfectly clean datasets

✅ **Uncertainty Quantification:**
- Provides confidence intervals for forecasts
- Uncertainty increases with forecast horizon

✅ **Fast Training:**
- Trains in 30-60 seconds on 30 days of hourly data
- Suitable for real-time web applications

#### Model Configuration

```python
Prophet(
    growth='linear',                # Linear trend (short-term weather changes)
    yearly_seasonality=True,        # Capture seasonal variations
    weekly_seasonality=True,        # Weekday/weekend patterns
    daily_seasonality=True,         # Day/night temperature cycles
    seasonality_mode='additive',    # Additive seasonal components
    changepoint_prior_scale=0.05,   # Moderate flexibility in trend changes
    interval_width=0.95,            # 95% confidence intervals
    uncertainty_samples=1000        # Monte Carlo samples for uncertainty
)
```

#### Custom Seasonalities

**Hourly Seasonality** (added for better intra-day patterns):
```python
model.add_seasonality(
    name='hourly',
    period=1,          # 1-day period
    fourier_order=8    # 8 Fourier terms for smooth curves
)
```

---

### 2. SARIMA (Seasonal ARIMA)

**Type:** Statistical time series model
**Status:** ⚠️ Implemented but disabled by default (slower training)
**Library:** `statsmodels >= 0.14.0`, `pmdarima >= 2.0.4`

#### What is SARIMA?

SARIMA (Seasonal AutoRegressive Integrated Moving Average) is a classical statistical model for time series forecasting that extends ARIMA with seasonal components.

**Model Components:**
- **AR (p):** AutoRegressive terms - dependency on past values
- **I (d):** Integration order - differencing to achieve stationarity
- **MA (q):** Moving Average - dependency on past forecast errors
- **S (P, D, Q, m):** Seasonal components with period m

#### Why SARIMA?

✅ **Statistical Baseline:**
- Well-established methodology
- Interpretable parameters
- Good for comparison against ML models

✅ **No External Dependencies:**
- Purely time-series based
- Doesn't require feature engineering

⚠️ **Limitations:**
- Slower training (2-5 minutes with auto_arima)
- Requires stationary data
- Less flexible than Prophet for multiple seasonalities

#### Auto-Parameter Selection

We use `pmdarima.auto_arima` to automatically find optimal parameters:

```python
auto_arima(
    ts_data,
    seasonal=True,
    m=24,                    # Daily seasonality (24 hours)
    max_p=3, max_q=3,       # AR and MA order limits
    max_P=2, max_Q=2,       # Seasonal AR and MA limits
    max_d=2, max_D=1,       # Differencing limits
    stepwise=True,           # Stepwise search (faster)
    trace=False,            # Don't print search progress
    n_fits=50               # Max models to try
)
```

---

## 🔄 Training Process

### Data Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  1. FETCH HISTORICAL DATA (Open-Meteo Archive API)     │
│     └─> 30 days × 24 hours = 720 hourly observations   │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  2. PREPARE TRAINING DATA                               │
│     • Convert to Prophet format (ds, y columns)         │
│     • Check for data quality issues                     │
│     • No missing value imputation (Prophet handles it)  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  3. TRAIN MODEL (with caching)                          │
│     • Hash training data                                │
│     • Check for cached model                            │
│     • If data changed or cache miss → retrain           │
│     • Save trained model to models_cache/               │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  4. GENERATE FORECAST                                   │
│     • Create future dataframe (168 hours = 7 days)      │
│     • Predict with confidence intervals                 │
│     • Return: datetime, temperature, lower, upper       │
└─────────────────────────────────────────────────────────┘
```

### Model Caching Strategy

**Purpose:** Avoid retraining on every request (saves 30-60 seconds per run)

**Implementation:**
1. **Data Hashing:** Generate hash of training data
2. **Cache Check:** Look for `models_cache/prophet_model.json`
3. **Validation:** Compare current data hash with cached metadata
4. **Decision:**
   - If hash matches → Load cached model (< 1 second)
   - If hash differs → Retrain and save new model

**Cache Files:**
```
models_cache/
├── prophet_model.json      # Serialized Prophet model
├── prophet_model.meta      # Metadata (hash, timestamp)
├── sarima_model.pkl        # Serialized SARIMA model
└── sarima_model.meta       # SARIMA metadata
```

**Manual Cache Clearing:**
```python
# In weather_forecast.py
ml_forecasts = generate_ml_forecasts(
    historical_df,
    force_retrain=True  # Force retraining, ignore cache
)
```

---

## 📈 Forecast Outputs

### Prophet Forecast DataFrame

| Column | Description | Example |
|--------|-------------|---------|
| `datetime` | Future timestamp | 2025-12-31 00:00:00 |
| `temperature_fahrenheit` | Point forecast | 42.5 |
| `lower_bound` | 95% CI lower bound | 38.2 |
| `upper_bound` | 95% CI upper bound | 46.8 |

### Confidence Intervals

**Prophet Uncertainty Methodology:**
- Uses Monte Carlo simulation (1000 samples)
- Incorporates:
  - Trend uncertainty
  - Seasonal component uncertainty
  - Observation noise
- Widens naturally with forecast horizon

**Typical Uncertainty:**
- **Next 24 hours:** ±2-3°F
- **3-4 days ahead:** ±4-5°F
- **7 days ahead:** ±6-8°F

---

## 🔍 Model Comparison

### Prophet vs Open-Meteo API

| Aspect | Prophet (ML) | Open-Meteo API (Professional) |
|--------|-------------|-------------------------------|
| **Training Data** | Historical temperature only | Numerical weather models (NWM) |
| **Input Features** | Time-based patterns | Atmospheric physics, pressure, humidity, wind |
| **Forecast Horizon** | 7 days | 7-16 days |
| **Update Frequency** | On-demand (cached) | Every 6 hours |
| **Strengths** | Pattern recognition, simplicity | Physics-based, comprehensive |
| **Limitations** | No weather events, limited by history | Requires complex infrastructure |
| **Typical Accuracy** | MAE: 3-5°F | MAE: 2-4°F |
| **Use Case** | Educational comparison, quick predictions | Production weather forecasts |

### When ML Forecasts Might Diverge

✅ **ML forecast may be better when:**
- Strong historical patterns repeat (e.g., clear seasonal trends)
- Recent weather has been typical/stable
- Short-term forecasts (1-3 days)

❌ **API forecast is usually better when:**
- Unusual weather events approaching (storms, fronts)
- Long-term forecasts (4-7 days)
- Rapidly changing conditions
- Access to real-time meteorological data

---

## ⚙️ Model Hyperparameters

### Prophet Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `growth` | linear | Short-term weather trends are approximately linear |
| `yearly_seasonality` | True | Capture seasonal temperature variations |
| `weekly_seasonality` | True | Weekday/weekend patterns |
| `daily_seasonality` | True | Day/night temperature cycles |
| `seasonality_mode` | additive | Temperature effects add together |
| `changepoint_prior_scale` | 0.05 | Moderate flexibility (0.001=rigid, 0.5=flexible) |
| `interval_width` | 0.95 | 95% confidence intervals |
| `uncertainty_samples` | 1000 | Monte Carlo samples for CI |

### SARIMA Auto-Selection

| Parameter | Range | Meaning |
|-----------|-------|---------|
| `p` (AR order) | 0-3 | How many past values to consider |
| `d` (Differencing) | 0-2 | Orders of differencing for stationarity |
| `q` (MA order) | 0-3 | How many past errors to consider |
| `P` (Seasonal AR) | 0-2 | Seasonal autoregressive terms |
| `D` (Seasonal diff) | 0-1 | Seasonal differencing |
| `Q` (Seasonal MA) | 0-2 | Seasonal moving average terms |
| `m` (Seasonality) | 24 | Period (24 hours for daily patterns) |

**Selected by:** AIC (Akaike Information Criterion) minimization

---

## 📊 Evaluation Metrics

### Metrics Calculated

**RMSE (Root Mean Squared Error):**
```python
RMSE = sqrt(mean((actual - predicted)²))
```
- Penalizes large errors more heavily
- Same unit as temperature (°F)
- Typical range: 2-6°F

**MAE (Mean Absolute Error):**
```python
MAE = mean(|actual - predicted|)
```
- Average absolute difference
- More interpretable than RMSE
- Typical range: 2-5°F

**MAPE (Mean Absolute Percentage Error):**
```python
MAPE = mean(|actual - predicted| / |actual|) × 100%
```
- Percentage error
- Useful for comparing different scales
- Typical range: 3-10%

**Correlation:**
```python
correlation = pearson_correlation(actual, predicted)
```
- Measures linear relationship (-1 to 1)
- > 0.9 = excellent agreement
- 0.7-0.9 = good agreement
- < 0.7 = poor agreement

### Limitations of Metrics

⚠️ **Important Caveats:**

1. **No Ground Truth for Future:** We can't truly validate forecasts until time passes
2. **Historical Performance ≠ Future Performance:** Past accuracy doesn't guarantee future accuracy
3. **Weather Events:** Sudden changes (storms, fronts) aren't captured by training data
4. **Data Quality:** Depends on accuracy of Open-Meteo's historical data

---

## 🚀 Usage Examples

### Python Script (weather_forecast.py)

```python
from weather_ml_forecast import generate_ml_forecasts

# Fetch historical data
historical_df = fetch_historical_data()

# Generate ML forecasts
ml_forecasts = generate_ml_forecasts(
    historical_df,
    forecast_periods=168,      # 7 days
    use_prophet=True,          # Enable Prophet
    use_sarima=False,          # Disable SARIMA (slow)
    force_retrain=False        # Use cache if available
)

# Access Prophet forecast
if 'prophet' in ml_forecasts:
    prophet_df = ml_forecasts['prophet']
    print(prophet_df[['datetime', 'temperature_fahrenheit', 'lower_bound', 'upper_bound']])
```

### Streamlit Dashboard (weather_app.py)

```python
# Generate forecasts in UI
with st.spinner('Generating ML forecasts...'):
    ml_forecasts = generate_ml_forecasts(
        historical_df,
        forecast_periods=168,
        use_prophet=True,
        use_sarima=False,
        force_retrain=False
    )

# Display comparison
if ml_forecasts and 'prophet' in ml_forecasts:
    compare_forecasts_ui(api_forecast_df, ml_forecasts['prophet'])
```

---

## 🎯 Interpretation Guide

### Reading the Dashboard

**Chart Colors:**
- 🔵 **Blue Line:** Historical actual temperatures (ground truth)
- 🔴 **Red Solid Line:** API forecast (Open-Meteo professional service)
- 🟢 **Green Dashed Line:** ML forecast (Prophet model)
- **Shaded Areas:** 95% confidence intervals

**What to Look For:**

1. **Agreement:** Do red and green lines track closely?
   - High correlation (>0.9) = models agree on trend
   - Low correlation (<0.7) = models see different patterns

2. **Bias:** Is one forecast consistently higher/lower?
   - ML warmer → Predicts higher temperatures
   - ML cooler → Predicts lower temperatures
   - Aligned (±1°F) → Similar predictions

3. **Uncertainty:** How wide are the confidence bands?
   - Narrow bands → Model is confident
   - Wide bands → High uncertainty
   - ML bands usually wider than API (less information)

4. **Divergence Points:** Where do forecasts differ most?
   - Often at changepoints (trend shifts)
   - Weather events the ML model can't see
   - Seasonal transitions

---

## ⚠️ Limitations & Assumptions

### Model Limitations

❌ **What Prophet CANNOT Do:**
1. **Predict Weather Events:** No knowledge of approaching storms, fronts, or systems
2. **Account for Atmospheric Physics:** Doesn't use pressure, humidity, wind data
3. **Handle Extreme Events:** Black swan weather events not in training data
4. **Real-time Updates:** Uses historical data only, no live observations
5. **Location-Specific Features:** Doesn't account for local geography, urban heat islands

❌ **What SARIMA CANNOT Do:**
1. **Multiple Complex Seasonalities:** Limited to one seasonal period
2. **Non-linear Relationships:** Assumes linear dependencies
3. **External Factors:** Purely time-series, no exogenous variables
4. **Long-term Forecasts:** Accuracy degrades quickly beyond 3-4 days

### Assumptions

📌 **Key Assumptions:**

1. **Stationarity (SARIMA):** Time series properties don't change dramatically
2. **Repeating Patterns:** Historical patterns will continue into future
3. **No Regime Changes:** Climate/weather patterns remain similar
4. **Data Quality:** Historical data is accurate and representative
5. **Sufficient Training Data:** 30 days is enough to capture patterns

### When NOT to Trust ML Forecasts

🚨 **Use API forecast instead when:**
- Storm or severe weather warnings are issued
- Unusual weather patterns are forming
- Planning critical outdoor activities (safety-related)
- Need forecasts beyond 3-4 days
- Seasonal transitions (autumn→winter)

✅ **ML forecasts are reasonable for:**
- Educational comparison and learning
- Stable weather periods
- Short-term planning (1-2 days)
- Understanding time series patterns
- Demonstrating ML capabilities

---

## 🔧 Troubleshooting

### Common Issues

**Problem:** Model training is slow (>2 minutes)

**Solutions:**
- ✅ Use cached models (default behavior)
- ✅ Disable SARIMA (`use_sarima=False`)
- ✅ Reduce `uncertainty_samples` in Prophet config
- ✅ Use fewer historical days for testing

---

**Problem:** "Prophet not installed" error

**Solution:**
```bash
pip install prophet
# On Windows, you may need:
conda install -c conda-forge prophet
```

---

**Problem:** Forecasts look unrealistic

**Possible Causes:**
- Insufficient training data (< 7 days)
- Data quality issues in historical data
- Unusual recent weather patterns
- Model caching issue (using outdated model)

**Solutions:**
- ✅ Check historical data quality
- ✅ Force retrain: `force_retrain=True`
- ✅ Clear cache: Delete `models_cache/` folder
- ✅ Increase `historical_days` in dashboard

---

**Problem:** Low correlation between forecasts

**Explanation:**
- This is often EXPECTED and not a problem
- Different models use different information
- ML sees patterns, API uses physics
- Divergence is informative (shows different approaches)

---

## 📚 Further Reading

### Prophet Documentation
- [Prophet Official Docs](https://facebook.github.io/prophet/)
- [Prophet Paper (Taylor & Letham, 2017)](https://peerj.com/preprints/3190/)
- [Forecasting at Scale](https://research.facebook.com/blog/2017/2/prophet-forecasting-at-scale/)

### SARIMA / Time Series
- [ARIMA Models (statsmodels)](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [pmdarima Documentation](http://alkaline-ml.com/pmdarima/)
- [Time Series Analysis Book (Hyndman & Athanasopoulos)](https://otexts.com/fpp3/)

### Weather Forecasting
- [Open-Meteo Documentation](https://open-meteo.com/en/docs)
- [Numerical Weather Prediction (NWP)](https://www.weather.gov/jetstream/nwp)
- [Ensemble Forecasting](https://www.ecmwf.int/en/about/media-centre/focus/2020/fact-sheet-ensemble-forecasting)

---

## 📞 Support & Contribution

### Report Issues
Found a problem with the ML forecasting models?
- Open an issue on [GitHub](https://github.com/jam1245/weather-forecast/issues)
- Include: error message, training data period, model parameters

### Contribute
Potential improvements welcome:
- Alternative ML models (LSTM, XGBoost, LightGBM)
- Feature engineering (adding weather variables)
- Ensemble forecasting (combining multiple models)
- Hyperparameter optimization
- Cross-validation and backtesting

---

**Version:** 1.0.0
**Last Updated:** December 2025
**Model Status:** Prophet ✅ | SARIMA ⚠️ (disabled by default)
