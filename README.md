# Diabetes-Prediction
This project focuses on building a complete web-based application that predicts the likelihood of diabetes in patients using machine learning. The system analyzes health metrics such as glucose level, BMI, blood pressure, age, and more to deliver accurate predictions.

# 🩺 Diabetes Prediction Web App

This is a Streamlit-based web application that predicts the likelihood of diabetes in patients using a trained machine learning model. The app takes key medical parameters as input and provides a prediction along with model interpretability using LIME.

## 🚀 Features

- ✅ User-friendly interface built with **Streamlit**
- 🧠 **Machine Learning Model** (Random Forest Classifier)
- 🪄 **Model Explainability** using **LIME**
- 📊 Input form for patient health metrics
- 📝 Instant prediction with a clear result
- 💡 Display of feature importance for better transparency

## 🧪 Tech Stack

- **Python**
- **Streamlit** – UI
- **Pandas, NumPy** – Data handling
- **Scikit-learn** – Machine Learning
- **LIME** – Model Explainability

## 📷 Screenshot

**Main Page**
![](https://github.com/ayushnegi127/diabetes-prediction/blob/main/Screenshot/ScreenShot 1.png)
**Explainable AI**
![](https://github.com/ayushnegi127/diabetes-prediction/blob/main/Screenshot/ScreenShot 2.png)

## 📌 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/ayushnegi127/diabetes-prediction.git
   cd diabetes-prediction
2. **Create a virtual environment and activate it**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate

4. **Install the dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py

