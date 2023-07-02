import streamlit as st 
import base64

st.header("Machine Learning Part 3")

pdf_file = "Hot100_ML_Part3.pdf"
with open(pdf_file,"rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

st.markdown(pdf_display, unsafe_allow_html=True)