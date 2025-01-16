import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.write("""
# Stock Price Prediction App
This app predicts the **Rental Price** based on some criteria!
""")

st.sidebar.header('User Input Parameters')

# Function to get user inputs
def user_input_features():
    BHK = st.sidebar.slider('Bedroom, Hallway, Kitchen', 1, 5, 2)
    Size = st.sidebar.slider('Size', 10, 4000000, 10000)
    Bathroom = st.sidebar.slider('Bathroom', 1, 10, 5)
    Area_Type = st.sidebar.selectbox('Area Type', ('Built Area', 'Carpet Area', 'Super Area'))
    City = st.sidebar.selectbox('City', ('Bangalore', 'Mumbai', 'Chennai', 'Hyderabad', 'Kolkata', 'Delhi'))
    Furnishing_Status = st.sidebar.selectbox('Furnishing Status', ('Semi-Furnished', 'Unfurnished', 'Furnished'))
    Tenant_Preferred = st.sidebar.selectbox('Tenant Preferred', ('Family', 'Bachelor', 'Bachelors/Family'))
    Point_of_Contact = st.sidebar.selectbox('Point of Contact', ('Agent', 'Owner', 'Builder'))
    Floors = st.sidebar.selectbox('The Floor Level of the Building', ('Ground', 'Lower', 'Upper', 'Other'))
    if Floors == 'Other':
        Floors = st.sidebar.slider('Floors', 1, 100, 1)
    else:
        Floors = Floors
    data = {
        'BHK': [BHK],
        'Size': [Size],
        'Bathroom': [Bathroom],
        'Area Type_Built Area': [1 if Area_Type == 'Built Area' else 0],
        'Area Type_Carpet Area': [1 if Area_Type == 'Carpet Area' else 0],
        'Area Type_Super Area': [1 if Area_Type == 'Super Area' else 0],
        'City_Bangalore': [1 if City == 'Bangalore' else 0],
        'City_Chennai': [1 if City == 'Chennai' else 0],
        'City_Delhi': [1 if City == 'Delhi' else 0],
        'City_Hyderabad': [1 if City == 'Hyderabad' else 0],
        'City_Kolkata': [1 if City == 'Kolkata' else 0],
        'City_Mumbai': [1 if City == 'Mumbai' else 0],
        'Furnishing Status_Furnished': [1 if Furnishing_Status == 'Furnished' else 0],
        'Furnishing Status_Semi-Furnished': [1 if Furnishing_Status == 'Semi-Furnished' else 0],
        'Furnishing Status_Unfurnished': [1 if Furnishing_Status == 'Unfurnished' else 0],     
        'Tenant Preferred_Bachelors': [1 if Tenant_Preferred == 'Bachelor' else 0],
        'Tenant Preferred_Bachelors/Family': [1 if Tenant_Preferred == 'Bachelors/Family' else 0],
        'Tenant Preferred_Family': [1 if Tenant_Preferred == 'Agent' else 0],
        'Point of Contact_Contact Agent': [1 if Tenant_Preferred == 'Company' else 0],
        'Point of Contact_Contact Builder': [1 if Point_of_Contact == 'Builder' else 0],
        'Point of Contact_Contact Owner': [1 if Point_of_Contact == 'Owner' else 0],
    }

    for i in range(1, 9):
        data[f'Floors_{i}'] = [1 if Floors == i else 0]
        for j in range(0, 10):
            if (i <= 7):
                data[f'Floors_{i}{j}'] = [1 if Floors == int(f"{i}{j}") else 0]
    
    data['Floors_80'] = [1 if Floors == 80 else 0]
    data['Floors_9'] = [1 if Floors == 9 else 0]
    data['Floors_Ground'] = [1 if Floors == 'Ground' else 0]
    data['Floors_Lower'] = [1 if Floors == 'Lower' else 0]
    data['Floors_Upper'] = [1 if Floors == 'Upper' else 0]
    
    data = pd.DataFrame(data)
    return data

# Get user input
df = user_input_features()

st.subheader('User Input Parameters')
st.dataframe(df, use_container_width=True, width=800)

# Load the saved model
model = pkl.load(open('model_predict.pkl', 'rb'))

# Predict using the model
prediction = np.expm1(model.predict(df))

st.subheader('Prediction')
st.write(f"The predicted rental price is: ${prediction[0]:,.2f}")   