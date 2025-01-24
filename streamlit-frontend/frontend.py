import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import hopsworks

# Set the style for plots and custom colors
sns.set(style="darkgrid")

# Initialize connection to Hopsworks Feature Store (using the class from your previous code)
class AQIFeatureStore:
    def __init__(self, project_name: str, api_key: str):
        self.project = hopsworks.login(project=project_name, api_key_value=api_key)
        self.fs = self.project.get_feature_store()
        self.feature_group_name = "aqi_features"
        self.version = 1

    def get_features(self):
        """Fetch features from the feature store."""
        feature_group = self.fs.get_feature_group(self.feature_group_name, version=self.version)
        features_df = feature_group.read()
        return features_df

# Replace load_data with fetching data from the feature store
def load_data():
    feature_store = AQIFeatureStore(
        project_name="AQI_PREDICTION_SYSTEM",
        api_key="YOUR-API-KEY"  # Replace with your actual API key
    )
    data = feature_store.get_features()
    return data

# Function to map AQI values to color scale
def aqi_to_color(aqi):
    if aqi <= 1:
        return '#2ecc71'  # Good (Green)
    elif aqi <= 2:
        return '#f39c12'  # Moderate (Yellow)
    elif aqi <= 3:
        return '#e67e22'  # Unhealthy for Sensitive Groups (Orange)
    elif aqi <= 4:
        return '#e74c3c'  # Unhealthy (Red)
    elif aqi <= 5:
        return '#9b59b6'  # Very Unhealthy (Purple)
    else:
        return '#c0392b'  # Hazardous (Maroon)

# Function to plot AQI prediction with added colors and styling based on weather
def plot_aqi(data):
    plt.figure(figsize=(10,6))
    color_map = [aqi_to_color(aqi) for aqi in data['predicted_aqi']]  # Apply color based on AQI values
    sns.lineplot(x='date', y='predicted_aqi', data=data, marker='o', palette=color_map, linewidth=2)

    plt.title('AQI Prediction Over Time', fontsize=16, color='white')
    plt.xlabel('Date', fontsize=14, color='white')
    plt.ylabel('Predicted AQI', fontsize=14, color='white')
    plt.xticks(rotation=45, color='white')
    plt.tight_layout()
    st.pyplot(plt)

# Function to plot AQI distribution (box plot) with color enhancements
def plot_aqi_distribution(data):
    plt.figure(figsize=(10,6))
    sns.boxplot(x=data['predicted_aqi'], color='#f39c12')
    
    plt.title('Distribution of Predicted AQI', fontsize=16, color='white')
    plt.xlabel('Predicted AQI', fontsize=14, color='white')
    plt.tight_layout()
    st.pyplot(plt)

# Function to display data and analysis
def display_data(data, city_filter=None):
    # If city filter is applied, filter the data
    if city_filter:
        data = data[data['city'].str.contains(city_filter, case=False)]
    
    # Display the raw data
    st.subheader("Raw AQI Prediction Data", anchor="raw-data")
    st.write(data)

    # Display AQI plot
    plot_aqi(data)

# Function to manipulate the data (filtering, viewing top rows)
def manipulate_data(data):
    st.subheader("Manipulate and View Data", anchor="manipulate-data")
    
    # Show top rows of the dataframe
    num_rows = st.slider('Select number of rows to display', 1, len(data), 5)
    st.write(data.head(num_rows))
    
    # Filter by AQI range
    min_aqi, max_aqi = st.slider('Select AQI range', int(data['predicted_aqi'].min()), int(data['predicted_aqi'].max()), (1, 5))
    filtered_data = data[(data['predicted_aqi'] >= min_aqi) & (data['predicted_aqi'] <= max_aqi)]
    st.write(f"Filtered data by AQI range ({min_aqi}-{max_aqi}):")
    st.write(filtered_data)

# Create the Streamlit app
def main():
    # Page title and theme settings
    st.set_page_config(page_title="AQI Prediction Dashboard", page_icon=":guardsman:", layout="wide")
    
    # Apply dark background color globally
    st.markdown(
        """
        <style>
        body {
            background-color: #2c3e50;
            color: white;
        }
        .stButton>button {
            background-color: #34495e;
            color: white;
        }
        .stSidebar {
            background-color: #2c3e50;
            color: white;
        }
        .stSidebar * {
            color: white !important;
        }
        .stTextInput>div>input {
            color: white;
            background-color: #34495e;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.title('AQI Prediction Dashboard')

    # Load data from Hopsworks Feature Store
    try:
        data = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Sidebar with city selection
    st.sidebar.title("City Selector")
    city_option = st.sidebar.radio("Choose a city", ["Islamabad", "Karachi", "Lahore", "All Cities"])

    if city_option != "All Cities":
        display_data(data, city_filter=city_option)
    else:
        display_data(data)

    # Show manipulation options
    st.sidebar.title("Data Manipulation")
    if st.sidebar.button('Manipulate Data'):
        manipulate_data(data)

    # Additional plot for AQI distribution (box plot)
    st.subheader("Additional Plot: AQI Distribution")
    plot_aqi_distribution(data)

    # Show a more detailed view of predictions for specific dates
    st.subheader("Detailed AQI Prediction for a Specific Date")
    date = st.selectbox("Select a Date", data['date'].unique())
    
    selected_data = data[data['date'] == date]
    st.write(f"Details for {date}:")
    st.write(selected_data[['city', 'date', 'max_temp_celsius', 'min_temp_celsius', 'mean_temp_celsius', 'predicted_aqi']])

    # Add some basic stats
    st.subheader("General Statistics on AQI")
    st.write(f"Mean Predicted AQI: {data['predicted_aqi'].mean():.2f}")
    st.write(f"Max Predicted AQI: {data['predicted_aqi'].max():.2f}")
    st.write(f"Min Predicted AQI: {data['predicted_aqi'].min():.2f}")

    # Customize background based on AQI level (optional visual cue)
    aqi_level = data['predicted_aqi'].mean()
    if aqi_level <= 2:
        st.markdown(f"<style>body {{background-color: #1abc9c;}}</style>", unsafe_allow_html=True)
    elif aqi_level <= 3:
        st.markdown(f"<style>body {{background-color: #f39c12;}}</style>", unsafe_allow_html=True)
    elif aqi_level <= 4:
        st.markdown(f"<style>body {{background-color: #e67e22;}}</style>", unsafe_allow_html=True)
    else:
        st.markdown(f"<style>body {{background-color: #e74c3c;}}</style>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
