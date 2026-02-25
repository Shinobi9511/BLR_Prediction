# 🏡 Bangalore House Price Prediction – Streamlit App

A Machine Learning powered web application that predicts **Bangalore house prices** based on property features such as:

* Total Square Feet
* Number of Bathrooms
* BHK (Bedrooms, Hall, Kitchen)
* Location (One-Hot Encoded)

The application is built using **Streamlit** and a trained **XGBoost regression model**.

---

## 🚀 Live Demo

Deployed on Streamlit Cloud ('https://blrprediction-8hzhhaam8kmxzfdp8mmzef.streamlit.app/')

---

## 📌 Project Overview

This project uses a supervised regression model to estimate house prices in Bangalore. The application:

* Loads a pre-trained `xgboost_model.pkl`
* Accepts user input via an interactive UI
* Applies preprocessing with one-hot encoding
* Returns predicted price in Lakhs (₹)

The feature structure strictly matches the model training pipeline to ensure consistency.

---

## 🛠️ Tech Stack

* Python 3.9+
* Streamlit
* XGBoost
* Pandas
* NumPy
* Joblib

---

## 📂 Project Structure

```
├── app.py                  # Streamlit application
├── xgboost_model.pkl       # Trained ML model
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/bangalore-house-price-prediction.git
cd bangalore-house-price-prediction
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application Locally

```bash
streamlit run app.py
```

The app will open in your browser at:

```
http://localhost:8501
```

---

## 🧠 Model Details

* Algorithm: XGBoost Regressor
* Problem Type: Regression
* Target Variable: House Price (in Lakhs ₹)
* Encoding: Manual One-Hot Encoding for location
* Numeric Features:

  * total_sqft
  * bath
  * bhk

The feature column list is hardcoded to exactly match the training dataset schema to prevent inference-time mismatch.

---

### User Inputs:

* Total Square Feet (Slider)
* Bathrooms (Slider)
* BHK (Slider)
* Location (Dropdown with 'other' option)

### Output:

* Predicted House Price in ₹ Lakhs
* Error handling for invalid input
* Informational disclaimer

---

## 🧩 Key Implementation Highlights

### ✅ Model Loading Optimization

Uses:

```python
@st.cache_resource
```

to prevent reloading the model on every interaction.

### ✅ Robust Preprocessing

* Initializes all feature columns to zero
* Assigns numeric inputs
* Applies location-based one-hot encoding
* Ensures strict column order consistency

### ✅ Error Handling

Graceful exception handling during prediction phase.

---

## ☁️ Streamlit Cloud Deployment

1. Push project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect repository
4. Select `app.py`
5. Deploy

Make sure:

* `xgboost_model.pkl` is present
* `requirements.txt` includes all dependencies

---

## 📦 Example requirements.txt

```
streamlit
pandas
numpy
joblib
xgboost
```

---

## ⚠️ Disclaimer

This application provides **estimated predictions** based on historical data. Actual property prices may vary due to:

* Market fluctuations
* Property condition
* Legal status
* Negotiation factors

---

## 👨‍💻 Author

Aanjney Kumawat
Petroleum Engineer | Data Science & ML Enthusiast
Skilled in Python, SQL, Tableau, and ML Deployment

---
