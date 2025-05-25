import streamlit as st
st.set_page_config(page_title="Diabetes Prediction App", layout="centered", page_icon="🧑‍⚕️")
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use('Agg')

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    model = joblib.load("rf_diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_scaler()


@st.cache_resource
def load_data_for_lime():
    df = pd.read_csv("diabetes.csv")
    X = df.drop("Outcome", axis=1)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X, X_scaled_df

X_original, X_scaled_df = load_data_for_lime()

st.title("Diabetes Prediction")

tab1, tab2 = st.tabs(["Predict", "Explainable AI"])

with tab1:
    st.header("Enter Patient Data")
    with st.form("prediction_form"):
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
        BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
        Age = st.number_input("Age", min_value=1, max_value=120, value=30)

        submit = st.form_submit_button("🔍 Predict")

    if submit:
        if Glucose == 0 or BloodPressure == 0 or BMI == 0:
            st.error("Glucose, Blood Pressure, and BMI cannot be zero. Please enter valid values.")
        else:
            input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                                        BMI, DiabetesPedigreeFunction, Age]],
                                      columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0][1] * 100

            if prediction == 1:
                st.error(f"⚠️ High risk of Diabetes ({proba:.2f}% probability). Please consult a doctor.")
            else:
                st.success(f"✅ Low risk of Diabetes ({proba:.2f}% probability). Keep maintaining a healthy lifestyle!")

            st.session_state['input_scaled'] = input_scaled
            st.session_state['input_data'] = input_data

with tab2:
    

    if "input_scaled" in st.session_state and "input_data" in st.session_state:
        input_scaled = st.session_state['input_scaled']
        input_data = st.session_state['input_data']

       
        input_row = input_scaled[0] 

        
        explainer = LimeTabularExplainer(
            training_data=X_scaled_df.values,
            feature_names=X_scaled_df.columns.tolist(),
            class_names=["Non-Diabetic", "Diabetic"],
            mode="classification"
        )

        
        explanation = explainer.explain_instance(
            data_row=input_row,
            predict_fn=model.predict_proba,
            num_features=8
        )

        
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)

        
        st.subheader("Patient Input Summary")
        st.dataframe(input_data)

    else:
        st.warning("⚠️ Please make a prediction in the first tab to see the LIME explanation here.")
