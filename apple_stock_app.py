import streamlit as st
import pandas as pd
import pickle as pkl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.write("""
# Stock Price Prediction App
This app predicts the **Stock Price** when the market closes!
""")

st.sidebar.header('User Input Parameters')

# Function to get user inputs
def user_input_features():
    High = st.sidebar.number_input('Highest Price', 1, 400, 23)
    Low = st.sidebar.number_input('Lowest Price', -10, 400, 21)
    Open = st.sidebar.number_input('Open Price', -10, 400, 22)
    data = pd.DataFrame({
        'Open': [Open],
        'High': [High],
        'Low': [Low],
    })
    return data

# Get user input
df = user_input_features()

st.subheader('User Input Parameters')
st.write(df.style.set_properties(**{'text-align': 'center'}))

# Load the saved model
stock = pkl.load(open('model_predict.pkl', 'rb'))

# Predict using the model
prediction = stock.predict(df)

st.subheader('Prediction')
st.write(f"Predicted Close Price: {prediction[0]:.2f}")

# Accuracy Comparison (Optional)
st.subheader('Upload a Test Dataset for Accuracy Comparison')
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Load uploaded dataset
    test_data = pd.read_csv(uploaded_file)

    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    if all(col in test_data.columns for col in required_columns):
        # Separate features and actual values
        X_test = test_data[['Open', 'High', 'Low']]
        y_actual = test_data['Close']

        # Make predictions
        y_pred = stock.predict(X_test)

        # Calculate metrics
        mae = (mean_absolute_error(y_actual, y_pred))*100
        r2 = (r2_score(y_actual, y_pred))*100

        # Display metrics
        st.subheader('Model Accuracy Metrics')
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}%")
        st.write(f"Root Squared (R^2): {r2:.4f}%")

        # Display the graph
        st.subheader('Stock Price Graph')
        st.write("This graph shows the stock price over time.")
        st.line_chart(test_data['Close'])

    else:
        st.error("The uploaded dataset must contain the following columns: Open, High, Low, Volume, Close")

