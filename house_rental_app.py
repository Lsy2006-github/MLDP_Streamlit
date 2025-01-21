import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

st.write("""
# Rental Price Prediction App
This app predicts the **Rental Price** of a property in India based on its features!
""")

st.write("Start to predict the rental price by selecting the features on the left sidebar!")

# Function to get user inputs
def user_input_features(BHK, Size, Bathroom, Area_Type, City, Furnishing_Status, Tenant_Preferred, Point_of_Contact, Floors):
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

# Sidebar - User input features
st.sidebar.header('User Input Parameters')
st.sidebar.write('Result are in Indian Rupees')

pred_btn, reset_btn = st.columns(2)

with pred_btn:
    predict_button = st.sidebar.button('Predict')
with reset_btn:
    if st.sidebar.button('Reset'):
        st.cache_data.clear()

BHK = st.sidebar.slider('Bedroom, Hallway, Kitchen', 1, 5, 2, disabled=predict_button)
Size = st.sidebar.number_input('Size', 10, 4000000, 10000, 100, disabled=predict_button)
Bathroom = st.sidebar.slider('Bathroom', 1, 10, 5, disabled=predict_button)
Area_Type = st.sidebar.selectbox('Area Type', ('Built Area', 'Carpet Area', 'Super Area'), disabled=predict_button)
City = st.sidebar.selectbox('City', ('Bangalore', 'Mumbai', 'Chennai', 'Hyderabad', 'Kolkata', 'Delhi'), disabled=predict_button)
Furnishing_Status = st.sidebar.selectbox('Furnishing Status', ('Semi-Furnished', 'Unfurnished', 'Furnished'), disabled=predict_button)
Tenant_Preferred = st.sidebar.selectbox('Tenant Preferred', ('Family', 'Bachelor', 'Bachelors/Family'), disabled=predict_button)
Point_of_Contact = st.sidebar.selectbox('Point of Contact', ('Agent', 'Owner', 'Builder'), disabled=predict_button)
Floors = st.sidebar.selectbox('The Floor Level of the Building', ('Ground', 'Lower', 'Upper', 'Other'), disabled=predict_button)
if Floors == 'Other':
    Floors = st.sidebar.slider('Floors', 1, 80, 10, disabled=predict_button)
else:
    Floors = Floors

# Add a button to trigger the input feature function
if predict_button:
    latest_iteration = st.empty()
    bar = st.progress(0)
    # Get user input
    df = user_input_features(BHK, Size, Bathroom, Area_Type, City, Furnishing_Status, Tenant_Preferred, Point_of_Contact, Floors)

    for i in range(51):
        # Update the progress bar with each iteration.
        bar.progress(i * 2, text=f'Predicting... {i * 2}%')
        time.sleep(0.1)
        if i == 50:
            bar.progress(100, text='Prediction Completed!')
            time.sleep(0.1)
        
    st.divider()
    # Show the user input parameters
    st.subheader('User Input Parameters')
    # Transpose the dataframe to show columns as rows
    df_transposed = df.T
    df_transposed.columns = ["User Input Parameters"]
    # Filter out the rows with 0 values
    filtered_df = df_transposed[df_transposed["User Input Parameters"] != 0]
    # Show the filtered dataframe
    st.dataframe(filtered_df, use_container_width=True)

    # Load the saved model
    model = joblib.load(open('model_predict.pkl', 'rb'))

    # Predict using the model
    prediction = np.expm1(model.predict(df))

    st.subheader('Prediction')
    st.write(f"The predicted rental price is: ${prediction[0]:,.2f} in Indian Rupees")
