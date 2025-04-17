import streamlit as st
import pickle
from numpy import exp
lr = pickle.load(open('../artifacts/model.pkl', 'rb'))
ohe = pickle.load(open('../artifacts/ohe.pkl', 'rb'))
st.title("Predict Insurance Charges")
age      = st.number_input("Age" , min_value = 18.0, max_value = 64.0)
child   = st.number_input("Children" , min_value = 0, max_value = 5)
bmi      = st.number_input("BMI")
smoker   = st.selectbox("is smoker?", ("yes", "no"),)
sex      = st.selectbox("Sex", ("male", "female"),)
region     = st.selectbox("region", ("southeast", "northwest", "southwest", "northeast"),)
temp = ohe.transform([[sex,smoker,region]]).toarray()
data = [age,bmi,child]
data.extend(temp[0])
pred = lr.predict([data])
st.text(exp(pred))


    