import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb 
from geopy.distance import geodesic
import pydeck as pdk
import numpy as np
from sklearn.preprocessing import MinMaxScaler

try:
    model = joblib.load(r"C:\Users\adiwi\OneDrive\Documents\CreditCardFraud copy\CreditCardFraud copy\fraud_detection_model.jb")
    encoder = joblib.load(r"C:\Users\adiwi\OneDrive\Documents\CreditCardFraud copy\CreditCardFraud copy\label_encoders.jb")
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.stop()

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("🔍 Credit Card Fraud Detection System")
st.write("Enter the transaction details below:")

# Use columns for cleaner layout
col1, col2 = st.columns(2)
with col1:
    merchant = st.selectbox("🏪 Merchant Name", list(encoder['merchant'].classes_))
    category = st.selectbox("📦 Category", list(encoder['category'].classes_))
    amt = st.number_input("💰 Transaction Amount", min_value=0.0, format="%.2f")
    lat = st.number_input("📍 User Latitude", format="%.6f")
    long = st.number_input("📍 User Longitude", format="%.6f")
with col2:
    merch_lat = st.number_input("🏬 Merchant Latitude", format="%.6f")
    merch_long = st.number_input("🏬 Merchant Longitude", format="%.6f")
    hour = st.slider("⏰ Transaction Hour", 0, 23, 12)
    day = st.slider("📅 Transaction Day", 1, 31, 15)
    month = st.slider("📆 Transaction Month", 1, 12, 6)
    gender = st.selectbox("🧑 Gender", ["Male", "Female"])
    cc_num = st.text_input("💳 Credit Card Number (hashed internally)")

distance = geodesic((lat, long), (merch_lat, merch_long)).km

if st.checkbox("Show Transaction Map"):
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(latitude=lat, longitude=long, zoom=10),
        layers=[
            pdk.Layer("ScatterplotLayer",
                      data=pd.DataFrame({'lat': [lat, merch_lat], 'lon': [long, merch_long]}),
                      get_position='[lon, lat]',
                      get_color='[200, 30, 0, 160]',
                      get_radius=100)
        ]
    ))

st.caption("🔐 Your credit card number is hashed and not stored.")

if st.button("🚨 Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        result = "🛑 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.subheader(f"Prediction: {result}")
        st.progress(proba)
        st.write(f"📊 Model Confidence (Fraud): {proba:.2%}")
    else:
        st.error("❗ Please fill all required fields.")
