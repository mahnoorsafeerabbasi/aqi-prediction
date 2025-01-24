import hopsworks
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AQIFeatureStore:
    def __init__(self, project_name: str, api_key: str):
        """Initialize connection to Hopsworks Feature Store"""
        self.project = hopsworks.login(project=project_name, api_key_value=api_key)
        self.fs = self.project.get_feature_store()
        
        # Define feature group metadata
        self.version = 1
        self.feature_group_name = "aqi_features"  # One feature group to store all features

    def create_feature_group(self):
        """Create or get a single feature group for weather, pollution, and predictions"""
        feature_group = self._create_combined_feature_group()
        return feature_group

    def _create_combined_feature_group(self):
        """Create a single feature group combining weather, pollution, and AQI prediction features"""
        feature_group = self.fs.get_or_create_feature_group(
            name=self.feature_group_name,
            version=self.version,
            description="Combined weather, pollution, and AQI prediction features",
            primary_key=['city', 'date'],
            event_time='date',
            online_enabled=True
        )
        return feature_group

    def preprocess_features(self, forecast_df: pd.DataFrame):
        """Preprocess and combine features into one dataframe"""
        # Ensure 'Date' is in datetime format
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'], errors='coerce').dt.date

        # Print the column names to debug and check if the column exists
        logging.info(f"Columns in forecast_df: {forecast_df.columns.tolist()}")

        # Weather features
        weather_features = forecast_df[[ 
            'City', 'Date', 'Mean Temp (°C)', 'Max Wind Speed (10m) (km/h)',
            'Dominant Wind Direction (°)', 'Shortwave Radiation Sum (MJ/m²)'
        ]].copy()

        # Clean the weather feature column names
        weather_features.columns = [
            col.lower()
            .replace(' ', '_')
            .replace('(°c)', ' ')
            .replace('(km/h)', ' ')
            .replace('(°)', ' ')
            .replace('(mj/m²)', 'mj_m2')
            .replace('(10m)', '10m')
            .replace('(', '')
            .replace(')', '')
            for col in weather_features.columns
        ]

        # Print the cleaned column names to check
        logging.info(f"Cleaned weather feature columns: {weather_features.columns.tolist()}")

        # Check if the column 'dominant_wind_direction_' exists
        if 'dominant_wind_direction_' in weather_features.columns:
            # Convert Dominant Wind Direction to integer and cast to 'bigint' (int64)
            weather_features['dominant_wind_direction_'] = weather_features['dominant_wind_direction_'].round().astype('int64')
        else:
            logging.warning("Column 'dominant_wind_direction_' not found in the weather features.")

        # Pollution features
        pollution_features = forecast_df[[ 
            'City', 'Date', 'PM10', 'PM2.5', 'NO2', 'SO2', 'CO', 'O3'
        ]].copy()

        # Clean the pollution feature column names
        pollution_features.columns = [col.lower().replace('.', '') for col in pollution_features.columns]

        # Predictions
        predictions = forecast_df[['City', 'Date', 'Predicted AQI']].copy()
        predictions.columns = [col.lower().replace(' ', '_') for col in predictions.columns]

        # Combine all features into one dataframe
        combined_features = weather_features.merge(pollution_features, on=['city', 'date'], how='left')
        combined_features = combined_features.merge(predictions, on=['city', 'date'], how='left')

        return combined_features

    def insert_features(self, forecast_df: pd.DataFrame):
        """Insert features into the feature group"""
        try:
            # Wait for available slot if necessary
            self.wait_for_available_slot()

            # Create or get feature group
            feature_group = self.create_feature_group()
            
            # Preprocess and combine features
            combined_features = self.preprocess_features(forecast_df)
            
            # Insert features into the feature group
            feature_group.insert(combined_features)
            
            logging.info("Successfully inserted features into feature group")
        
        except Exception as e:
            logging.error(f"Error inserting features: {str(e)}")
            raise

    def check_active_jobs(self):
        """Check the number of active jobs in the project"""
        try:
            jobs_api = self.project.get_jobs_api()
            jobs = jobs_api.get_jobs()
            # Most Hopsworks versions use 'execution_status' instead of 'state'
            active_jobs = [job for job in jobs if getattr(job, 'execution_status', None) == 'RUNNING']
            return len(active_jobs)
        except Exception as e:
            logging.error(f"Error checking active jobs: {str(e)}")
            return 0

    def wait_for_available_slot(self):
        """Wait for a job slot to become available if the maximum parallel executions are reached"""
        while self.check_active_jobs() >= 5:
            logging.info("Parallel job limit reached, waiting for available slot...")
            time.sleep(60)  # Wait for 1 minute before checking again

    def create_feature_view(self, name: str, version: int = 1):
        """Create a feature view using the combined feature group"""
        try:
            # Get the feature group
            feature_group = self.fs.get_feature_group(self.feature_group_name, version=self.version)
            
            # Create feature view (can be a simple selection of all features)
            feature_view = self.fs.create_feature_view(
                name=name,
                version=version,
                query=feature_group.select_all(),
                description="Combined feature view for weather, pollution, and AQI predictions"
            )
            
            return feature_view
        except Exception as e:
            logging.error(f"Error creating feature view: {str(e)}")
            raise


def main():
    # Initialize feature store
    feature_store = AQIFeatureStore(
        project_name="AQI_PREDICTION_SYSTEM",
        api_key="YOUR-API-KEY"  # Replace with your actual API key
    )
    
    # Load forecast data
    forecast_df = pd.read_csv(
        Path('data') / f'forecast_data_with_predictions_{datetime.now().strftime("%Y%m%d")}.csv'
    )
    
    # Insert features into feature store
    feature_store.insert_features(forecast_df)
    
    # Create feature view
    feature_view = feature_store.create_feature_view(
        name="aqi_prediction_features",
        version=1
    )
    
    logging.info("Successfully created feature view")

if __name__ == "__main__":
    main()
