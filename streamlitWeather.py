import streamlit as st
import streamlit_folium as folium
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import geocoder
import joblib
import pandas as pd
import numpy as np

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Get user's location using geocoder
user_location = geocoder.ip('me')


def predict_using_localization():
# Check if location information is available
    if user_location.latlng:
        latitude, longitude = user_location.latlng

        # Reverse geocode to get location name
        location = geocoder.osm([latitude, longitude], method='reverse')
        location_name = location.address if location else "Unknown Location"
        st.write("Your current location:", location_name)
        # Make sure all required weather variables are listed here
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "evapotranspiration", "et0_fao_evapotranspiration","wind_speed_10m"],
            "wind_speed_unit": "ms"
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_data = {
            "date": pd.to_datetime(hourly.Time(), unit="s"),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "dew_point_2m": hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(5).ValuesAsNumpy(),
        }

        hourly_dataframe = pd.DataFrame(data=hourly_data)

        # Display the dataframe in the Streamlit app
        st.write("Hourly Data:", hourly_dataframe)

        # Calculate daily average temperature and relative humidity
        daily_avg_temperature = hourly_dataframe.groupby(hourly_dataframe['date'].dt.date)['temperature_2m'].mean()
        daily_avg_relative_humidity = hourly_dataframe.groupby(hourly_dataframe['date'].dt.date)['relative_humidity_2m'].mean()
        daily_avg_wind_speed = hourly_dataframe.groupby(hourly_dataframe['date'].dt.date)['wind_speed_10m'].mean()
        # Display daily averages
        #st.write("Daily Average Temperature:", daily_avg_temperature)
        #st.write("Daily Average Relative Humidity:", daily_avg_relative_humidity)
        #st.write("Daily Average Wind Speed:", daily_avg_wind_speed)
        X_Pred = pd.DataFrame({
            'moy_Température[°C]': [daily_avg_temperature[0]],  # Add your rayonnement_solaire value here
            'moy_Vitesse du vent[m/s]':[daily_avg_wind_speed[0]],
            'moy_Humidité Relative[%]': [daily_avg_relative_humidity[0]]  # Add your humidite_relative value here
        })
        model = joblib.load('model_3_params_temps_vitesse_humRel.joblib')
        # Make predictions using the model
        y_pred = model.predict(X_Pred)

        st.write(f"Predicted ET0 for {hourly_dataframe['date'].dt.date[0]}: {y_pred[0]}")
        st.write()

    else:
        st.warning("Location information not available.")

class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
def make_prediction(moy_temperature, moy_wind_speed,moy_rayonnement_solaire, moy_relative_humidity):
    modelParams=[]
    # Create a DataFrame for prediction using user's parameters
    X_Pred_params = pd.DataFrame()
    if(moy_temperature!=None):
        X_Pred_params['moy_Température[°C]']=[moy_temperature]
        modelParams.append("temps")
    if(moy_wind_speed!=None):
        X_Pred_params['moy_Vitesse du vent[m/s]']=[moy_wind_speed]
        modelParams.append("vitesse")
    if(moy_rayonnement_solaire!=None):
        X_Pred_params['moy_Rayonnement solaire[W/m2]']=[moy_rayonnement_solaire]
        modelParams.append("raySol")
    if(moy_relative_humidity!=None):
        X_Pred_params['moy_Humidité Relative[%]']=[moy_relative_humidity]
        modelParams.append("humRel")
    modelParamNames=""
    for elem in modelParams:
        modelParamNames+="_"+elem
    #model_3_params_vitesse_raySol_humRel.joblib
    if len(modelParams)>=2:
        modelName="model_"+str(len(modelParams))+"_params"+modelParamNames+".joblib"
        #st.write(modelName)
        model = joblib.load(modelName)
        # Make predictions using the model
        y_pred_params = model.predict(X_Pred_params)
        return y_pred_params[0]
    else:
        st.warning("Please enter at least 2 parameters ")
        return 0
    

def predict_using_parameters():
    # Input fields for user's parameters
    moy_temperature = st.number_input("Enter La Température[°C]:",value=None)
    moy_wind_speed = st.number_input("Enter La Vitesse du Vent[m/s]:", min_value=0.0,value=None)
    moy_rayonnement_solaire = st.number_input("Enter Le Rayonnement solaire[W/m2]:", min_value=0.0,value=None)
    moy_relative_humidity = st.number_input("Enter L\'Humidité Relative[%]:", min_value=0.0, max_value=100.0,value=None)
   
    # Predict button
    if st.button("Predict"):
        # Call the cached function to get the prediction result
        prediction_result = make_prediction(moy_temperature, moy_wind_speed,moy_rayonnement_solaire, moy_relative_humidity)

        st.write(f"Predicted ET0: {prediction_result}")
        st.write()

# Main Streamlit app
st.title("ET0 Prediction App")


# Define the tabs
tabs = ["Predict using my localisation", "Predict using my parameters"]
selected_tab = st.sidebar.radio("Select a prediction method", tabs)

# Display content based on the selected tab
if selected_tab == "Predict using my localisation":
    st.header("Prediction Using User's Localisation")
    predict_using_localization()
else:
    st.header("Prediction Using User's Parameters")
    predict_using_parameters()

