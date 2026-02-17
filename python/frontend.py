import streamlit as st
import joblib
import pandas as pd

# Load model
model_package = joblib.load("../models/car_price_model_v2.pkl")
model = model_package["model"]

st.set_page_config(
    page_title="🚗 Car Price Predictor",
    layout="wide",
    page_icon="🚗"
)

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>🚗 Interactive Car Price Predictor</h1>
<h4 style='text-align: center;'>Built by Hemanth Kumar | B.Tech CSE ML Project</h4>
<hr>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("📊 About Project")
st.sidebar.info("""
This ML model predicts used car price  
using features:

• Age  
• KM Driven  
• Weight  
• Horse Power  

End-to-End ML Project with FastAPI + Streamlit
""")

st.sidebar.success("Model: Random Forest")

# ---------- INPUT SECTION ----------
st.subheader("🔧 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Car Age (years)", 0, 20, 5)
    km = st.number_input("KM Driven", 0, 300000, 40000)

with col2:
    weight = st.number_input("Car Weight (kg)", 500, 3000, 1200)
    hp = st.slider("Horse Power", 50, 300, 100)

st.write("")

# ---------- PREDICT BUTTON ----------
if st.button("💰 Predict Price"):

    df = pd.DataFrame([[age, km, weight, hp]],
                      columns=["Age", "KM", "Weight", "HP"])

    price = model.predict(df)[0]

    st.markdown(f"""
    <div style='background-color:#D4EFDF;padding:20px;border-radius:10px'>
        <h2 style='color:#1E8449;text-align:center'>
        Estimated Car Price = ₹ {round(price,2)}
        </h2>
    </div>
    """, unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown("""
<hr>
<center>
Made with ❤️ using Machine Learning & Streamlit
</center>
""", unsafe_allow_html=True)
