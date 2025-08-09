# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# # Load the trained model
# model = tf.keras.models.load_model("model.h5")

# # load the encoder and scaler
# with open("E:\Project_ANN_deep_learning(customer will leave bank or not)\onehot_encoder_geo.pkl","rb") as file:
#     onehot_encoder_geo = pickle.load(file)

# with open("E:\Project_ANN_deep_learning(customer will leave bank or not)\label_encoder_gender.pkl","rb") as file:
#     label_encoder_gender = pickle.load(file)

# with open("E:\Project_ANN_deep_learning(customer will leave bank or not)\scaler.pkl","rb") as file:
#     scaler = pickle.load(file) 

# # Streamlit app
# st.title("Customer Churn Prediction")

# # User input
# geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
# gender = st.selectbox("Gender",label_encoder_gender.classes_)
# age = st.slider("Age",18,92)
# balance = st.number_input("Balance")
# credit_score = st.number_input("Credit Score")
# estimated_salary = st.number_input("Estimated Salary")
# tenure = st.slider("Tenure",0,10)
# num_of_products = st.slider("Number of Products",1,4)
# has_cr_card = st.selectbox("Has Credit Card", [0,1])
# is_active_member = st.selectbox("Is Active Member", [0,1])

# # Prepare the input data
# input_data = pd.DataFrame({
#     "CreditScore": [credit_score],
#     "Gender": [label_encoder_gender.transform([gender])[0]],
#     "Age": [age],
#     "Tenure": [tenure],
#     "Balance": [balance],
#     "NumOfProducts": [num_of_products],
#     "HasCrCard": [has_cr_card],
#     "IsActiveMember": [is_active_member],
#     "EstimatedSalary": [estimated_salary]
       
# })

# # Onehot encode "Geography" column
# geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(["Geography"]))

# # Combine one-hot encoded colimns with input data
# input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)

# # Predict churn
# prediction = model.predict(input_data_scaled)
# prediction_prob = prediction[0][0]

# if prediction_prob > 0.5:
#     print("The customer is likely to churn")
# else:
#     print("The customer is not likely to churn")

import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Load encoders and scaler
with open(r"E:\Project_ANN_deep_learning(customer will leave bank or not)\onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open(r"E:\Project_ANN_deep_learning(customer will leave bank or not)\label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open(r"E:\Project_ANN_deep_learning(customer will leave bank or not)\scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title("Customer Churn Prediction")

geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

# Get feature names safely for all sklearn versions
try:
    geo_columns = onehot_encoder_geo.get_feature_names_out(["Geography"])
except AttributeError:
    try:
        geo_columns = onehot_encoder_geo.get_feature_names(["Geography"])
    except AttributeError:
        geo_columns = [f"Geography_{cat}" for cat in onehot_encoder_geo.categories_[0]]

geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)

# Combine and scale
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display
if prediction_prob > 0.5:
    st.error(f"⚠️ The customer is likely to churn (probability: {prediction_prob:.2%})")
else:
    st.success(f"✅ The customer is not likely to churn (probability: {1 - prediction_prob:.2%})")
