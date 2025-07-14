
import streamlit as st
import pandas as pd
import numpy as np
import joblib

#rf_model = joblib.load("rf_model.pkl")
svr_model = joblib.load("svr_model.pkl")
scaler = joblib.load("scaler.pkl")

Device_id = st.number_input('Device_id')
co2_1 = st.number_input('co2_1')
V1  = st.number_input('V1')



inputs = {'Device_id' : Device_id , "co2_1" : co2_1, "V1" : V1}

inputs = pd.DataFrame(inputs, index=['Device_id','co_2','V1'])
preprocessed_inputs = scaler.transform(inputs)

if st.button("Predict"):
    output = svr_model.predict(preprocessed_inputs)


    st.write(f"Prediction : {output[0]}")