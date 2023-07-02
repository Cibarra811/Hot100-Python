import streamlit as st 
import plotly.express as px 
import pickle

df = pickle.load['ml_df.pkl']

col1, col2 = st.columns(2)

x_axis_val = col1.selectbox('Select the X-axis', options=df.columns)
y_axis_val = col2.selectbox('Select the Y-axis', options=df.columns)

plot = px.scatter(df, x=x_axis_val, y=y_axis_val)
st.plotly_chart(plot, use_container_width=True)

