

import streamlit as st
import joblib
import numpy as np

scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title("Predição de Churn de Clientes")

st.divider()

st.write("Preencha os dados abaixo para prever se um cliente irá cancelar o serviço:")

st.divider()

gender = st.selectbox("Entra com o genero do cliente:", ["Masculino", "Femenino"])

age = st.number_input("Entra com a idade do cliente:", min_value=13, max_value=100, value=30)

tenure = st.number_input("Entra com o tempo de permanência em meses do cliente:", min_value=0, max_value=130, value=10)

monthly_charges = st.number_input("Entra com o valor mensal do cliente:", min_value=30, max_value=150)

st.divider()

prediction_button = st.button("Executar Predição")

if prediction_button:

    selecao_genero = 1 if gender == "Femenino" else 0
    input_data = np.array([[selecao_genero, age, tenure, monthly_charges]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("O cliente provavelmente irá cancelar o serviço (Churn).")
    else:
        st.info("O cliente provavelmente irá permanecer (Não Churn).")