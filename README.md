# 🏠 Bangalore House Price Prediction App

A Machine Learning web application built using **Streamlit** that predicts house prices in Bangalore based on:

- Total Square Feet
- Number of Bathrooms
- BHK (Bedrooms, Hall, Kitchen)
- Location (One-Hot Encoded)

The model was trained using **Linear Regression (scikit-learn)** and deployed using Streamlit.

---

## 🚀 Live Demo

()

---

## 📌 Features

- Interactive sidebar inputs
- Dynamic location selection
- Automatic feature alignment with trained model
- Error handling for prediction issues
- Clean and responsive UI
- Production-ready structure

---

## 🧠 Machine Learning Model

- Algorithm: Linear Regression
- Framework: scikit-learn
- Encoding: One-Hot Encoding for location
- Input Features:
  - `total_sqft`
  - `bath`
  - `bhk`
  - Location columns (generated during training)

The app dynamically loads model feature names using:

```python
model.feature_names_in_
