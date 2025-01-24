import requests
import pandas as pd
import logging
from datetime import datetime, timedelta, date
from pathlib import Path
from sklearn.impute import SimpleImputer
import joblib

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# City coordinates
cities = {
    'Karachi': {'lat': 24.8607, 'lon': 67.0011},
    'Islamabad': {'lat': 33.6844, 'lon': 73.0479},
    'Lahore': {'lat': 31.558, 'lon': 74.3507}
}

# Air pollution API key
API_KEY_AIR = 'YOUR-API-KEY'

# Weather forecast API URL
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast?"

# Air pollution forecast API URL
AIR_POLLUTION_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast?"

# Load the saved AQI prediction model and scaler
best_aqi_model = joblib.load('best_aqi_model_GradientBoosting.joblib')
scaler = joblib.load('scaler_for_best_aqi_model.joblib')

# Feature columns used during training
feature_columns = [
    'PM10', 'PM2.5', 'NO2', 'SO2', 'CO', 'O3', 
    'Mean Temp (°C)', 'Max Wind Speed (10m) (km/h)', 
    'Dominant Wind Direction (°)', 'Shortwave Radiation Sum (MJ/m²)'
]

def fetch_forecast_weather_data(city, coords):
    """Fetch weather forecast data for the next 3 days"""
    url = (
        f"{WEATHER_API_URL}"
        f"latitude={coords['lat']}&longitude={coords['lon']}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,"
        f"sunrise,sunset,wind_speed_10m_max,wind_gusts_10m_max,"
        f"wind_direction_10m_dominant,shortwave_radiation_sum"
        f"&timezone=Asia/Karachi"
    )
    logging.info(f"Fetching weather forecast for {city}")
    response = requests.get(url)
    return response

def fetch_forecast_air_pollution_data(city, coords):
    """Fetch forecast air pollution data for the next 3 days"""
    url = (
        f"{AIR_POLLUTION_API_URL}lat={coords['lat']}&lon={coords['lon']}&appid={API_KEY_AIR}"
    )
    logging.info(f"Fetching air pollution forecast for {city}")
    response = requests.get(url)
    return response

def process_forecast_weather_response(city, response, weather_data):
    """Process weather forecast API response and add to weather_data list"""
    if response.status_code == 200:
        data = response.json()
        for i, day in enumerate(data['daily']['time']):
            weather_data.append({
                'City': city,
                'Date': pd.to_datetime(day).date(),
                'Max Temp (°C)': data['daily']['temperature_2m_max'][i],
                'Min Temp (°C)': data['daily']['temperature_2m_min'][i],
                'Mean Temp (°C)': data['daily']['temperature_2m_mean'][i],
                'Max Apparent Temp (°C)': data['daily']['apparent_temperature_max'][i],
                'Min Apparent Temp (°C)': data['daily']['apparent_temperature_min'][i],
                'Mean Apparent Temp (°C)': data['daily']['apparent_temperature_mean'][i],
                'Sunrise': data['daily']['sunrise'][i],
                'Sunset': data['daily']['sunset'][i],
                'Max Wind Speed (10m) (km/h)': data['daily']['wind_speed_10m_max'][i],
                'Max Wind Gusts (10m) (km/h)': data['daily']['wind_gusts_10m_max'][i],
                'Dominant Wind Direction (°)': data['daily']['wind_direction_10m_dominant'][i],
                'Shortwave Radiation Sum (MJ/m²)': data['daily']['shortwave_radiation_sum'][i],
            })
    else:
        logging.error(f"Error fetching weather forecast data for {city}: {response.status_code}")
        logging.error(f"Response: {response.text}")

def process_forecast_air_pollution_response(city, response, air_pollution_data):
    """Process air pollution forecast API response and add to air_pollution_data list"""
    if response.status_code == 200:
        data = response.json()
        for i, record in enumerate(data['list']):
            record_date = datetime.utcfromtimestamp(record['dt']).date()
            air_pollution_data.append({
                'City': city,
                'Date': record_date,
                'PM10': record['components'].get('pm10', None),
                'PM2.5': record['components'].get('pm2_5', None),
                'NO2': record['components'].get('no2', None),
                'SO2': record['components'].get('so2', None),
                'CO': record['components'].get('co', None),
                'O3': record['components'].get('o3', None),
                'AQI': record['main'].get('aqi', None)
            })
    else:
        logging.error(f"Error fetching air pollution forecast data for {city}: {response.status_code}")
        logging.error(f"Response: {response.text}")

def fetch_forecast_data():
    """Fetch both weather and air pollution forecast data for the next 3 days"""
    all_weather_data = []
    all_air_pollution_data = []
    
    for city, coords in cities.items():
        # Fetch weather forecast data
        weather_response = fetch_forecast_weather_data(city, coords)
        process_forecast_weather_response(city, weather_response, all_weather_data)
        
        # Fetch air pollution forecast data
        air_pollution_response = fetch_forecast_air_pollution_data(city, coords)
        process_forecast_air_pollution_response(city, air_pollution_response, all_air_pollution_data)
    
    weather_df = pd.DataFrame(all_weather_data)
    air_pollution_df = pd.DataFrame(all_air_pollution_data)
    
    if not weather_df.empty and not air_pollution_df.empty:
        # Select only numeric columns to group by and aggregate
        numeric_columns_weather = weather_df.select_dtypes(include=['number']).columns
        numeric_columns_air_pollution = air_pollution_df.select_dtypes(include=['number']).columns

        # Group by City and Date, then take the mean of each column (excluding non-numeric)
        weather_df = weather_df.groupby(['City', 'Date'], as_index=False)[numeric_columns_weather].mean()
        air_pollution_df = air_pollution_df.groupby(['City', 'Date'], as_index=False)[numeric_columns_air_pollution].mean()

        # Merge datasets
        forecast_df = pd.merge(weather_df, air_pollution_df, on=['City', 'Date'], how='inner')
        forecast_df = forecast_df.sort_values(['City', 'Date'])
        logging.info(f"Forecast data collected: {len(forecast_df)} records")
        return forecast_df
    else:
        logging.error("Failed to collect forecast data!")
        return pd.DataFrame()


def predict_future_aqi(forecast_df):
    """Predict AQI for the next 3 days using the weather and air pollution forecast data"""
    if forecast_df.empty:
        logging.error("Empty forecast data!")
        return pd.DataFrame()

    # Prepare features (using relevant columns)
    features = forecast_df[feature_columns]
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features)
    
    # Scale the features using the previously fitted scaler
    features_scaled = scaler.transform(features_imputed)
    
    # Predict AQI using the pre-trained model
    future_aqi_predictions = best_aqi_model.predict(features_scaled)
    
    # Add predicted AQI to the forecast dataframe
    forecast_df['Predicted AQI'] = future_aqi_predictions
    
    return forecast_df

def main():
    """Main function to fetch forecast data and predict AQI for the next 3 days"""
    logging.info("Starting forecast data collection for the next 3 days")
    
    # Fetch forecast data (weather and air pollution)
    forecast_df = fetch_forecast_data()
    
    if not forecast_df.empty:
        # Predict AQI
        forecast_with_predictions = predict_future_aqi(forecast_df)
        
        if not forecast_with_predictions.empty:
            # Save the forecast data with predictions
            output_dir = Path('data')
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f'forecast_data_with_predictions_{datetime.now().strftime("%Y%m%d")}.csv'
            forecast_with_predictions.to_csv(output_path, index=False)
            logging.info(f"Successfully saved forecast data with predictions to {output_path}")
        else:
            logging.error("Failed to make AQI predictions.")
    else:
        logging.error("No forecast data collected!")
    
if __name__ == "__main__":
    main()
