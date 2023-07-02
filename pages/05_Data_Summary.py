import streamlit as st 
import pandas  as pd
import numpy as np

df = pd.read_pickle('ml_df.pkl')

st.title("DataFrame")

st.write(df.head(50))

st.header('Statistics of DataFrame')
st.write(df.describe())
st.session_state['df'] = df