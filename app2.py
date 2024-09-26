import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso model
with open('quizvivek.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a title for your app
st.title('Monthly Revenue Prediction App')

# Create input fields for the features
st.header('Enter Store Details')

average_order_value = st.number_input('Average Order Value')
total_orders = st.number_input('Total Orders')
customer_lifetime_value = st.number_input('Customer Lifetime Value')
average_order_frequency = st.number_input('Average Order Frequency')


# Create a button to trigger the prediction
if st.button('Predict Monthly Revenue'):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'average_order_value': [average_order_value],
        'total_orders': [total_orders],
        'customer_lifetime_value': [customer_lifetime_value],
        'average_order_frequency': [average_order_frequency],
    })
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Display the prediction
    st.header('Predicted Monthly Revenue')
    st.write(f'{prediction:.2f}')

