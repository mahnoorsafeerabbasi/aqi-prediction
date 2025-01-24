import hopsworks

# Connect to Hopsworks
project_name = "AQI_PREDICTION_SYSTEM"  # Replace with your actual project name
api_key = "w1IOG2L7FLfYy52R.4xG7KWAtnOnmn3jzyS0CNnB68jR1kX7qYpZN3GDzJ1ifLReBzUke4howinfecLD3"  # Replace with your actual API key

project = hopsworks.login(project=project_name, api_key_value=api_key)

# Get the Model Registry
mr = project.get_model_registry()

# Upload the Gradient Boosting model
gb_model = mr.python.create_model("aqi_gradient_boosting")
gb_model.save("best_aqi_model_GradientBoosting.joblib")

# Upload the scaler model
scaler_model = mr.python.create_model("aqi_scaler")
scaler_model.save("scaler_for_best_aqi_model.joblib")
