import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo entrenado
#with open('modelo_opt_rf_m.pkl', 'rb') as file:
#    #modelo = pickle.load(file)
modelo = joblib.load('modelo_opt_rf_m.pkl')
# Definir la interfaz de usuario en Streamlit
st.title('Predicción de Precios de Laptops')

# Controles de entrada para las características
# 'GHz', 'Ram', 'screen_width', 'screen_height','Inches']
ghz = st.number_input('GHz del CPU', min_value=0.1, max_value=5.0, value=2.5)
ram = st.number_input('RAM (GB)', min_value=1, max_value=64, value=8)
screen_width = st.number_input('Ancho de Pantalla', min_value=800, max_value=4000, value=1920)
screen_height = st.number_input('Alto de Pantalla', min_value=600, max_value=3000, value=1080)
inches = st.number_input('Inches (GB)', min_value=10.5, max_value=20.5, value=13.3)

type_gaming = st.selectbox('¿Es Gaming?', ['No', 'Sí'])
type_notebook = st.selectbox('¿Es Notebook?', ['No', 'Sí'])

# Convertir entradas a formato numérico
type_gaming = 1 if type_gaming == 'Sí' else 0
type_notebook = 1 if type_notebook == 'Sí' else 0

# Botón para realizar predicción
if st.button('Predecir Precio'):
    # Crear DataFrame con las entradas
    #input_data = pd.DataFrame([[ghz, ram, screen_width, screen_height, inches,  type_gaming, type_notebook]],
    #                columns=['GHz','Ram', 'screen_width', 'screen_height', 'Inches','TypeName_Gaming', 'TypeName_Notebook'])
    input_data = pd.DataFrame([[ghz, ram, screen_width, screen_height, inches]],
                    columns=['GHz','Ram', 'screen_width', 'screen_height', 'Inches'])

    # Estandarización de las características
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_data)

    # Realizar predicción
    prediction = modelo.predict(input_scaled)

    # Mostrar predicción
    st.write(f'Precio predecido: {prediction[0]:.2f} euros')



    # Mostrar predicción
    st.write(f'Precio predecido: {prediction[0]:.2f} euros')


