import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#Loading the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scalers
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender= pickle.load(f) 

with open('onehot_encoder_geo.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)

##Streamlit app
st.title('Customer Leave Prediction')

#User Input
geography = st.selectbox('Geography' , label_encoder_geo.categories_[0])
gender = st.selectbox('Gender' , label_encoder_gender.classes_)
age = st.slider('Age' , 18 , 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure' , 0 , 10)
num_of_products = st.slider('Number of Products' , 1, 4)
has_cr_card = st.selectbox('Has Credit Card' ,[0,1])
is_active_member = st.selectbox('Is Active Member' , [0,1])

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender' : [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#One-hot encode 'Geography'
geo_encoded = label_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

#Encode 'Gender'
input_data['Gender'] = label_encoder_gender.transform(input_data[['Gender']])

#Combine one-hot encoded columns with input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

#Drop original 'Geography' column (already one-hot encoded)
input_data = input_data.drop('Geography', axis=1)

#Scale the input data
input_scale = scaler.transform(input_data)

### Predict whether person will leave the bank or not
prediction=model.predict(input_scale)
prediction_probability = prediction[0][0]

st.write(f"LEAVE PROBABILITY: {prediction_probability: .2f} ")

if prediction_probability > 0.5:
    st.write("🔴 The customer is predicted to **LEAVE**")
else:
    st.write("🟢 The customer is **NOT likely to leave**")