import pandas as pd

import streamlit as st

import plotly.express as px


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


st.sidebar.header(
    "Filtros"
)
st.sidebar.title(
    "Panel de Control"
)
st.sidebar.write(
    "Seleccione los alumnos que desea analizar"
)
alumnos = st.sidebar.multiselect(
    "Selecciona alumnos",
    datos["Alumno"],
    default=datos["Alumno"]
)

tipo_grafico = st.sidebar.radio(
    "Tipo de gráfico",
    [
        "Barras",
        "Circular"
    ]
)
datos_filtrados = datos[
    datos["Alumno"].isin(
        alumnos
    )
]

st.sidebar.header(
"Exportación"
)

csv = datos_filtrados.to_csv(
    index=False
).encode("utf-8")

st.sidebar.download_button(
    label="📥 Descargar CSV",
    data=csv,
    file_name="alumnos_filtrados.csv",
    mime="text/csv"
)

st.header(
    "KPIs")

col1, col2, col3 = st.columns(3)

if len(datos_filtrados) > 0:

    col1.metric(
        "Nota media",
        round(
            datos_filtrados["Nota"].mean(),
            2
        )
    )

    col2.metric(
        "Alumnos seleccionados",
        len(datos_filtrados)
    )

    col3.metric(
        "Nota máxima",
        datos_filtrados["Nota"].max()
    )


    st.header(
        "Análisis"
    )
    col_tabla, col_grafico = st.columns(2)

    with col_tabla:
        st.subheader(
            "Datos"
            )

        st.dataframe(
            datos_filtrados
            )

    with col_grafico:
        st.subheader(
            "Gráfico"
            )
    if tipo_grafico == "Barras":

        fig = px.bar(
            datos_filtrados,
            x="Alumno",
            y="Nota",
            title="Notas de alumnos",
            color="Nota",
            text="Nota"
        )

    else:

        fig = px.pie(
            datos_filtrados,
            names="Alumno",
            values="Nota",
            title="Distribución de notas"
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.warning(
    "Seleccione al menos un alumno"
    )

if st.button(
        "Mostrar mensaje"
    ):
        
        st.success(
            "Botón pulsado correctamente 🚀"
        )