# app.py
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Explainable Generative AI — Demo", layout="wide")
st.title("Explainable Generative AI Dashboard — Demo")

@st.cache_data
def load_data():
    df = pd.read_csv("../02_Data/Processed/features.csv", parse_dates=['Date'])
    return df

df = load_data()
st.header("S&P500 (processed)")
st.line_chart(df.set_index('Date')['Adj Close'])

st.sidebar.header("Controls")
if st.sidebar.button("Generate sample forecast"):
    st.info("This will call the model and SHAP explainer (not yet implemented).")
