import streamlit as st
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "linear_regression_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

feature_columns = model.feature_names_in_

location_columns = [
    col for col in feature_columns
    if col not in ['total_sqft', 'bath', 'bhk']
]

st.set_page_config(page_title="Bangalore House Price Predictor")

st.title("🏠 Bangalore House Price Prediction")

with st.sidebar:
    st.header("House Details")

    total_sqft = st.slider(
        "Total Square Feet",
        min_value=300.0,
        max_value=10000.0,
        value=1200.0,
        step=50.0
    )

    bath = st.slider(
        "Number of Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    bhk = st.slider(
        "Number of BHK",
        min_value=1,
        max_value=10,
        value=2
    )

    location = st.selectbox("Location", sorted(location_columns))


if st.button("Predict Price"):

    try:
        
        input_df = pd.DataFrame(
            np.zeros((1, len(feature_columns))),
            columns=feature_columns
        )

        
        input_df['total_sqft'] = total_sqft
        input_df['bath'] = bath
        input_df['bhk'] = bhk

        
        input_df[location] = 1

        
        prediction = model.predict(input_df)[0]

        st.success(f"### 💰 Predicted Price: ₹ {prediction:.2f} Lakhs")

    except Exception as e:
        st.error("Prediction failed.")
        st.error(str(e))