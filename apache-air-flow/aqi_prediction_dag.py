import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.impute import SimpleImputer
import joblib
import hopsworks
from airflow import DAG
from airflow.operators.python import PythonOperator
from hopsworks import login

# Constants for API and Model Details
API_KEY_AIR = 'YOUR-API-KEY'
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast?"
AIR_POLLUTION_API_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast?"
FEATURE_COLUMNS = [
    'PM10', 'PM2.5', 'NO2', 'SO2', 'CO', 'O3', 
    'Mean Temp (°C)', 'Max Wind Speed (10m) (km/h)', 
    'Dominant Wind Direction (°)', 'Shortwave Radiation Sum (MJ/m²)'
]

cities = {
    'Karachi': {'lat': 24.8607, 'lon': 67.0011},
    'Islamabad': {'lat': 33.6844, 'lon': 73.0479},
    'Lahore': {'lat': 31.558, 'lon': 74.3507}
}

# Hopsworks project and API key
PROJECT_NAME = "AQI_PREDICTION_SYSTEM"
API_KEY = " Replace with your actual API key" 

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Connect to Hopsworks
project = hopsworks.login(project=PROJECT_NAME, api_key_value=API_KEY)
fs = project.get_feature_store()

# Load pre-trained models and scaler
# Load pre-trained models and scaler
model = joblib.load('/home/noor/airflow/dags/best_aqi_model_GradientBoosting.joblib')
scaler = joblib.load('/home/noor/airflow/dags/scaler_for_best_aqi_model.joblib')

# Define Feature Group
feature_group_name = "aqi_features"

def fetch_weather_data(city, coords):
    url = (
        f"{WEATHER_API_URL}latitude={coords['lat']}&longitude={coords['lon']}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"apparent_temperature_max,apparent_temperature_min,apparent_temperature_mean,"
        f"sunrise,sunset,wind_speed_10m_max,wind_gusts_10m_max,"
        f"wind_direction_10m_dominant,shortwave_radiation_sum"
        f"&timezone=Asia/Karachi"
    )
    logging.info(f"Fetching weather forecast for {city}")
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def fetch_air_pollution_data(city, coords):
    url = (
        f"{AIR_POLLUTION_API_URL}lat={coords['lat']}&lon={coords['lon']}&appid={API_KEY_AIR}"
    )
    logging.info(f"Fetching air pollution forecast for {city}")
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def process_forecast_data(cities):
    all_weather_data = []
    all_air_pollution_data = []
    
    for city, coords in cities.items():
        weather_data = fetch_weather_data(city, coords)
        air_pollution_data = fetch_air_pollution_data(city, coords)

        if weather_data and air_pollution_data:
            # Process and append data from weather and air pollution API
            for i, day in enumerate(weather_data['daily']['time']):
                all_weather_data.append({
                    'City': city,
                    'Date': pd.to_datetime(day).date(),
                    'Max Temp (°C)': weather_data['daily']['temperature_2m_max'][i],
                    'Min Temp (°C)': weather_data['daily']['temperature_2m_min'][i],
                    'Mean Temp (°C)': weather_data['daily']['temperature_2m_mean'][i],
                    'Max Apparent Temp (°C)': weather_data['daily']['apparent_temperature_max'][i],
                    'Min Apparent Temp (°C)': weather_data['daily']['apparent_temperature_min'][i],
                    'Mean Apparent Temp (°C)': weather_data['daily']['apparent_temperature_mean'][i],
                    'Max Wind Speed (10m) (km/h)': weather_data['daily']['wind_speed_10m_max'][i],
                    'Dominant Wind Direction (°)': weather_data['daily']['wind_direction_10m_dominant'][i],
                    'Shortwave Radiation Sum (MJ/m²)': weather_data['daily']['shortwave_radiation_sum'][i],
                })
            
            for i, record in enumerate(air_pollution_data['list']):
                record_date = datetime.utcfromtimestamp(record['dt']).date()
                all_air_pollution_data.append({
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
    
    # Combine weather and pollution data
    weather_df = pd.DataFrame(all_weather_data)
    pollution_df = pd.DataFrame(all_air_pollution_data)
    combined_df = pd.merge(weather_df, pollution_df, on=['City', 'Date'], how='inner')
    
    return combined_df

def train_and_predict(cities):
    forecast_df = process_forecast_data(cities)
    if not forecast_df.empty:
        features = forecast_df[FEATURE_COLUMNS]
        
        # Impute missing values and scale features
        imputer = SimpleImputer(strategy='mean')
        features_imputed = imputer.fit_transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # Predict AQI
        predictions = model.predict(features_scaled)
        forecast_df['Predicted AQI'] = predictions
        
        return forecast_df
    else:
        logging.error("No forecast data found!")
        return None

def insert_to_feature_store(forecast_df):
    try:
        # Create or retrieve feature group
        feature_group = fs.get_or_create_feature_group(
            name=feature_group_name,
            version=1,
            description="Air quality prediction features",
            primary_key=['City', 'Date'],
            event_time="Date"
        )

        # Insert features into the feature group
        feature_group.insert(forecast_df)
        logging.info("Successfully inserted features into feature group")
    except Exception as e:
        logging.error(f"Error inserting features: {str(e)}")

def aqi_prediction_pipeline():
    forecast_df = train_and_predict(cities)
    if forecast_df is not None:
        insert_to_feature_store(forecast_df)

# Airflow DAG setup
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'aqi_prediction_dag',
    default_args=default_args,
    description='Air Quality Index prediction pipeline',
    schedule_interval=timedelta(hours=24),  # Run daily
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    
    run_pipeline = PythonOperator(
        task_id='run_aqi_prediction_pipeline',
        python_callable=aqi_prediction_pipeline,
        dag=dag,
    )

    run_pipeline
