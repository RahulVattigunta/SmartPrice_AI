# SmartPrice AI: Mobile Cost Predictor and Classifier

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Step 2: Load Dataset (local fallback)
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/dphi-official/Datasets/master/mobile_price_classification.csv"
        data = pd.read_csv(url)
    except:
        st.warning("⚠️ Online dataset not accessible. Loading backup CSV file...")
        data = pd.read_csv("test.csv")  # Place this file locally in the project folder

    data['price'] = (data['ram'] * 3 + data['battery_power'] * 0.5 + data['px_height'] * 0.2 + data['px_width'] * 0.2 + data['mobile_wt'] * -0.1).astype(int)
    bins = [0, 10000, 25000, 50000]
    labels = ['Budget', 'Mid-Range', 'Premium']
    data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels)
    return data

data = load_data()

# Step 3: Feature Preparation
X = data.drop(['price', 'price_category'], axis=1)
y_reg = data['price']
y_cls = data['price_category']

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_cls, test_size=0.2, random_state=42)

# Step 4: Train Models
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_r, y_train_r)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_c, y_train_c)

# Step 5: Streamlit UI
st.title("📱 SmartPrice AI - Mobile Price Predictor & Classifier")
st.markdown("Enter mobile specifications to get predicted price and category.")

# Feature Inputs
ram = st.slider("RAM (MB)", 500, 8000, 2000)
battery = st.slider("Battery Power (mAh)", 500, 5000, 2000)
px_height = st.slider("Pixel Height", 0, 2000, 800)
px_width = st.slider("Pixel Width", 0, 2000, 800)
mobile_wt = st.slider("Mobile Weight (grams)", 80, 250, 150)

# Get dynamic input with all features
input_dict = {
    'ram': ram,
    'battery_power': battery,
    'px_height': px_height,
    'px_width': px_width,
    'mobile_wt': mobile_wt
}

# Fill remaining features with 0s
for col in X.columns:
    if col not in input_dict:
        input_dict[col] = 0

# Ensure correct feature order
input_df = pd.DataFrame([[input_dict[col] for col in X.columns]], columns=X.columns)
input_scaled = scaler.transform(input_df)

# Predict
pred_price = regressor.predict(input_scaled)[0]
pred_category = classifier.predict(input_scaled)[0]

# Display results
st.subheader("🔍 Predicted Results")
st.write(f"**Predicted Price:** ₹{int(pred_price)}")
st.write(f"**Price Category:** {pred_category}")

# Optional Metrics
if st.checkbox("Show Model Evaluation Metrics"):
    y_pred_r = regressor.predict(X_test_r)
    y_pred_c = classifier.predict(X_test_c)
    st.markdown("### Regression Metrics")
    st.write("MAE:", mean_absolute_error(y_test_r, y_pred_r))
    st.write("RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_r)))

    st.markdown("### Classification Metrics")
    st.write("Accuracy:", accuracy_score(y_test_c, y_pred_c))
    st.text("Classification Report:\n" + classification_report(y_test_c, y_pred_c))
