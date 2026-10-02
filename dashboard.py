import pandas as pd
import numpy as np

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

st.title(
    "Dashboard Estudiantes"
)

datos = pd.DataFrame(
    {
        "Alumno": [
            "Ana",
            "Luis",
            "Marta"
        ],
        "Nota": [
            8.5,
            7.2,
            9.1
        ]
    }
)

# Muestra este DataFrame en la página web.
st.dataframe(datos)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Nota media",
    round(
        datos["Nota"].mean(),
        2
    )
)

col2.metric(
    "Alumnos",
    len(datos)
)

col3.metric(
    "Nota máxima",
    datos["Nota"].max()
)

alumno = st.selectbox(
    "Selecciona alumno",
    datos["Alumno"]
)

datos_filtrados = datos[
    datos["Alumno"] == alumno
]

st.dataframe(
    datos_filtrados
)

if st.button(
    "Mostrar mensaje"
):
    
    st.success(
        "Botón pulsado correctamente 🚀"
    )

fig = px.bar(
    datos,
    x="Alumno",
    y="Nota",
    title="Notas por alumno"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

fig = px.bar(
    datos,
    x="Alumno",
    y="Nota",
    title="Notas por alumno"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
fig = px.bar(
    datos,
    x="Alumno",
    y="Nota",
    title="Notas por alumno"
)

st.plotly_chart(
    fig,
    use_container_width=True
)