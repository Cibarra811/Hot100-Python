import streamlit as st 
import pandas  as pd
import pickle

df = pd.read_csv('ml_df.csv')

st.title("DataFrame")

st.write(df.head(50))

st.header('Statistics of DataFrame')
st.write(df.describe())
st.session_state['df'] = df
