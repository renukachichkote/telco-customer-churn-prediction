import streamlit as st
import pandas as pd
import pickle
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'rfc_model.sav')
 
# Load the model
model = pickle.load(open(model_path, 'rb'))
 
# Load the data to get the columns
df1 = pd.read_csv('first_churn.csv')
 
# Streamlit app UI
st.title("Churn Prediction App")
 
# Input fields for each column
input_data = []
for col in df1.columns:
    value = st.text_input(f"Enter {col}:")
    input_data.append(value)
 
if st.button("Predict"):
    try:
        # Create a dictionary for input data
        input_dict = {col: [val] for col, val in zip(df1.columns, input_data)}
        
        # Create DataFrame for input data
        input_df = pd.DataFrame(input_dict)
 
        # Combine with original data for one-hot encoding
        df2 = pd.concat([df1, input_df], ignore_index=True)
        
        # Select columns for which dummies to be created
        columns_to_dummify = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method']
 
        # Create dummies for selected columns
        df2_dummies_subset = pd.get_dummies(df2[columns_to_dummify])
        
        # Concatenate the original input_df with the dummies subset
        df2_combined = pd.concat([df2.drop(columns=columns_to_dummify), df2_dummies_subset], axis=1)
        
        # Make prediction
        predictions = model.predict(df2_combined.tail(1))
        churn_probability = model.predict_proba(df2_combined.tail(1))[:, 1] * 100
 
        if predictions[0] == 1:
            st.write("This customer is likely to churn!!")
        else:
            st.write("This customer is likely to continue!!")
 
        st.write(f"Confidence: {churn_probability[0]:.2f}%")
 
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")