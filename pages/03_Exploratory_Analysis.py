import streamlit as st 
import base64

st.header("Exploratory Analysis Part 2")

pdf_file = "Hot100_Exploratory_Analysis_Part2.pdf"
with open(pdf_file,"rb") as f:
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')

pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

st.markdown(pdf_display, unsafe_allow_html=True)