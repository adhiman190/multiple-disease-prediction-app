import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import plotly.express as px
from fpdf import FPDF
from streamlit_option_menu import option_menu
import google.generativeai as genai

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Healthcare System",
    layout="wide",
    page_icon="🩺"
)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR,"models")
DATA_PATH = os.path.join(BASE_DIR,"data")

os.makedirs("database",exist_ok=True)

DB_PATH = "database/patients.db"

# -----------------------------
# DATABASE
# -----------------------------
conn = sqlite3.connect(DB_PATH,check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
name TEXT,
disease TEXT,
result TEXT
)
""")

conn.commit()

# -----------------------------
# MODEL LOADER
# -----------------------------
def load_model(name):

    path=os.path.join(MODEL_PATH,name)

    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.warning(f"Model not found: {name}")
        return None

# -----------------------------
# LOAD MODELS
# -----------------------------
diabetes_model=load_model("diabetes_model5.sav")
heart_model=load_model("heart_disease_model.sav")
lung_model=load_model("lung_cancer_model.sav")

# -----------------------------
# SAVE PREDICTION
# -----------------------------
def save_prediction(name,disease,result):

    cursor.execute(
        "INSERT INTO predictions VALUES (?,?,?)",
        (name,disease,result)
    )

    conn.commit()

# -----------------------------
# GENERATE PDF REPORT
# -----------------------------
def generate_pdf(name,disease,result):

    pdf=FPDF()

    pdf.add_page()
    pdf.set_font("Arial",size=14)

    pdf.cell(200,10,"AI Medical Report",ln=True)

    pdf.cell(200,10,f"Patient Name: {name}",ln=True)
    pdf.cell(200,10,f"Disease: {disease}",ln=True)
    pdf.cell(200,10,f"Prediction: {result}",ln=True)

    file="report.pdf"
    pdf.output(file)

    return file

# -----------------------------
# SIDEBAR MENU
# -----------------------------
with st.sidebar:

    selected=option_menu(
        "AI Healthcare System",
        [
            "Home",
            "Doctor Dashboard",
            "Diabetes Prediction",
            "Heart Disease Prediction",
            "Lung Cancer Prediction",
            "Prediction History",
            "AI Doctor"
        ],
        icons=[
            "house",
            "bar-chart",
            "activity",
            "heart",
            "lungs",
            "database",
            "chat"
        ]
    )

# -----------------------------
# HOME PAGE
# -----------------------------
if selected=="Home":

    st.title("🩺 AI Healthcare Disease Prediction System")

    st.write("""
This intelligent healthcare system predicts diseases using Machine Learning.

### Features
✔ Disease Prediction Models  
✔ Doctor Analytics Dashboard  
✔ Patient Prediction History  
✔ Downloadable Medical Reports  
✔ AI Medical Chatbot  
""")

# -----------------------------
# DOCTOR DASHBOARD
# -----------------------------
if selected=="Doctor Dashboard":

    st.title("📊 Doctor Analytics Dashboard")

    try:
        df=pd.read_csv(os.path.join(DATA_PATH,"lung_cancer.csv"))
    except:
        st.warning("Dataset not found")
        st.stop()

    col1,col2=st.columns(2)

    with col1:
        fig=px.histogram(
            df,
            x="AGE",
            color="LUNG_CANCER",
            title="Age Distribution vs Cancer"
        )
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        fig=px.pie(
            df,
            names="SMOKING",
            title="Smoking Percentage"
        )
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Correlation Heatmap")

    corr=df.corr(numeric_only=True)

    fig=px.imshow(corr,text_auto=True)

    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# DIABETES PREDICTION
# -----------------------------
if selected=="Diabetes Prediction":

    st.title("Diabetes Prediction")

    name=st.text_input("Patient Name")

    glucose=st.number_input("Glucose")
    bmi=st.number_input("BMI")

    if st.button("Predict Diabetes"):

        data=np.array([[glucose,bmi]])

        prediction=diabetes_model.predict(data)

        result="Diabetic" if prediction[0]==1 else "Not Diabetic"

        st.success(result)

        save_prediction(name,"Diabetes",result)

        file=generate_pdf(name,"Diabetes",result)

        with open(file,"rb") as f:
            st.download_button(
                "Download Medical Report",
                f,
                "report.pdf"
            )

# -----------------------------
# HEART DISEASE
# -----------------------------
if selected=="Heart Disease Prediction":

    st.title("Heart Disease Prediction")

    name=st.text_input("Patient Name")

    age=st.number_input("Age")
    chol=st.number_input("Cholesterol")

    if st.button("Predict Heart Disease"):

        data=np.array([[age,chol]])

        prediction=heart_model.predict(data)

        result="Heart Disease Detected" if prediction[0]==1 else "Healthy"

        st.success(result)

        save_prediction(name,"Heart Disease",result)

# -----------------------------
# LUNG CANCER
# -----------------------------
if selected=="Lung Cancer Prediction":

    st.title("Lung Cancer Prediction")

    name=st.text_input("Patient Name")

    age=st.number_input("Age")

    smoking=st.selectbox("Smoking",["No","Yes"])
    smoking=1 if smoking=="Yes" else 0

    if st.button("Predict Lung Cancer"):

        data=np.array([[age,smoking]])

        prediction=lung_model.predict(data)

        result="Cancer Risk" if prediction[0]==1 else "Low Risk"

        st.success(result)

        save_prediction(name,"Lung Cancer",result)

# -----------------------------
# PREDICTION HISTORY
# -----------------------------
if selected=="Prediction History":

    st.title("Patient Prediction History")

    df=pd.read_sql_query(
        "SELECT * FROM predictions",
        conn
    )

    st.dataframe(df)

# -----------------------------
# GEMINI AI DOCTOR
# -----------------------------
if selected=="AI Doctor":

    st.title("🤖 AI Medical Assistant")

    api_key=st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    question=st.text_input(
        "Ask a medical question"
    )

    if st.button("Ask Doctor"):

        genai.configure(api_key=api_key)

        model=genai.GenerativeModel("gemini-pro")

        response=model.generate_content(question)

        st.write(response.text)