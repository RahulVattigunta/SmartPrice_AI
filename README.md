# 📱 SmartPrice AI - Mobile Price Predictor & Classifier

## 📌 Overview
This project is a **machine learning-powered web app** built using **Streamlit** and **Scikit-learn**, featuring:

✅ Price prediction using regression 💸  
✅ Price category classification (Budget / Mid-Range / Premium) 🏷️  
✅ Streamlit UI for real-time predictions ⚙️  
✅ Interactive sliders for dynamic input 🎛️  
✅ Model evaluation metrics display 📊  
✅ Local fallback for dataset loading 📂  

---

## 📂 Project Structure
```
smartprice_ai/
│── smartprice_ai.py # Main Streamlit app
│── mobile_price_classification.csv # Optional local dataset
│── requirements.txt # Python dependencies
│── README.md # Project overview
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/smartprice-ai.git
cd smartprice-ai
```
### 2️⃣ Create and Activate Virtual Environment
```sh
python3 -m venv venv
source venv/bin/activate
```
### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```
### 4️⃣ Run the App
```sh
streamlit run smartprice_ai.py
```
Then visit http://localhost:8501 in your browser.

## 🔍 Features
🔢 Regression model for predicting mobile prices

🏷️ Classification model to assign price category

🧪 Model evaluation: MAE, RMSE, Accuracy, Classification Report

🧠 Uses RandomForestRegressor and RandomForestClassifier

⚠️ Fallback to local CSV if online dataset fails

💡 Visual UI powered by Streamlit sliders

## 🧠 ML Pipeline

StandardScaler for feature normalization

Train/Test split for evaluation

Random Forest models for robustness

Real-time UI to interact with trained models

## 🎯 Inputs
RAM (MB)

Battery Power (mAh)

Pixel Height

Pixel Width

Mobile Weight (grams)

These are used to compute a synthetic price and predict a class.

## 🏆 Contribution
Want to improve this app? Add more features?
Feel free to fork, submit PRs, or open issues 🚀

## 📜 License
This project is released under the MIT License.
Use it freely, learn from it, and extend it. 😄

## 👨‍💻 Author

Developed by **Rahul Vattigunta**  
Connect on [LinkedIn](https://www.linkedin.com/in/rahulvattigunta/) for collaboration or professional inquiries.

