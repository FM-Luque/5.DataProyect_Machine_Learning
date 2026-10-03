import pandas as pd
import streamlit as st

import plotly.express as px

st.title(
    "Dashboard EDA Real"
)

df = pd.read_csv(
   r"C:\Users\equipo\Desktop\THE POWER\DOCUMENTOS DE VIDEOS DIRECTOS\5.DataProyect_Machine_Learning\data\processed\03_datos_eda.csv"
)


# NUEVAS VARIABLES

df["grupo_edad"] = pd.cut(
    df["edad"],
    bins=[17,20,23,26,29],
    labels=[
        "18-20",
        "21-23",
        "24-26",
        "27-29"
    ]
)

df["grupo_sueno"] = pd.cut(
    df["horas_sueno"],
    bins=[4,6,8,10],
    labels=[
        "4-6h",
        "6-8h",
        "8-10h"
    ]
)


# ========================================
# FILTTOS ACADEMICOS
# ========================================

st.sidebar.header(
    "Filtros Académicos"
)

aprobado = st.sidebar.multiselect(
    "Aprobado",
    sorted(df["aprobado_sn"].unique()),
    default=sorted(df["aprobado_sn"].unique())
)

df_filtrado = df[
    df["aprobado_sn"].isin(aprobado)
]

st.sidebar.header(
    "Análisis"
)

variable_analisis = st.sidebar.selectbox(
    "Variable a analizar",
    [
        "nivel_dificultad",
        "horario_estudio_preferido",
        "estilo_aprendizaje",
        "tiene_tutor_sn",
        "rendimiento_academico",
        "grupo_edad",
        "grupo_sueno"
    ]
)

st.sidebar.header(
    "Visualización"
)

tipo_grafico = st.sidebar.radio(
    "Tipo gráfico",
    [
        "Barras",
        "Circular"
    ]
)

st.sidebar.header(
    "Análisis Bivariante"
)

variable_x = st.sidebar.selectbox(
    "Variable X",
    [
        "nivel_dificultad",
        "horario_estudio_preferido",
        "estilo_aprendizaje",
        "tiene_tutor_sn",
        "rendimiento_academico",
        "grupo_edad",
        "grupo_sueno"
    ]
)

variable_y = st.sidebar.selectbox(
    "Variable Y",
    [
        "nivel_dificultad",
        "horario_estudio_preferido",
        "estilo_aprendizaje",
        "tiene_tutor_sn",
        "rendimiento_academico",
        "grupo_edad",
        "grupo_sueno",
    ]
)

medida = st.sidebar.selectbox(
    "Métrica",
    [
        "Frecuencia",
        "Nota Media",
        "Horas Estudio",
        "Asistencia %"
    ]
)
# ========================================
# EXPORTACIÓN
# ========================================

st.sidebar.header(
    "Exportación"
)

csv = df_filtrado.to_csv(
    index=False
).encode("utf-8")

st.sidebar.download_button(
    label="📥 Descargar CSV",
    data=csv,
    file_name="datos_filtrados.csv",
    mime="text/csv"
)


# ========================================
# KPIs
# ========================================

st.header(
    "KPIs Generales"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Nota Media",
    round(
        df_filtrado["nota_final"].mean(),
        2
    )
)

col2.metric(
    "Aprobación %",
    round(
        df_filtrado["aprobado_ml"].mean() * 100,
        2
    )
)

col3.metric(
    "Horas Estudio",
    round(
        df_filtrado["horas_estudio_semanal"].mean(),
        2
    )
)

col4.metric(
    "Asistencia %",
    round(
        df_filtrado["tasa_asistencia"].mean(),
        2
    )
)


# ========================================
# ANALISIS GRAFICOS
# ========================================

st.header(
    "Análisis Global"
)

df_analisis = (
    df_filtrado
    .groupby(variable_analisis)
    .size()
    .reset_index(name="total")
)

if tipo_grafico == "Barras":

    fig = px.bar(
        df_analisis,
        x=variable_analisis,
        y="total",
        text="total",
        color=variable_analisis,
        title=f"Distribución por {variable_analisis}"
    )
else:

    fig = px.pie(
        df_analisis,
        names=variable_analisis,
        values="total",
        title=f"Distribución por {variable_analisis}"
    )

st.plotly_chart(
    fig,
    use_container_width=True
)

# ========================================
# ANALISIS BIVARIANTES
# ========================================


st.header(
    "Análisis Bivariantes"
)

if variable_x == variable_y:

    st.warning(
        "Seleccione variables diferentes."
    )

elif len(df_filtrado) > 0:

    st.info(
        f"Tasa de aprobados entre {variable_x} y {variable_y}"
    )

    tabla_aprobacion = (
        pd.pivot_table(
            df_filtrado,
            values="aprobado_ml",
            index=variable_x,
            columns=variable_y,
            aggfunc="mean"
        ) * 100
    ).round(2)

    st.dataframe(
        tabla_aprobacion
    )

    st.info(
        f"{medida} entre {variable_x} y {variable_y}"
    )


if variable_x == variable_y:

    st.warning(
        "Seleccione variables diferentes."
    )

elif len(df_filtrado) > 0:

    if medida == "Frecuencia":

        tabla_bivariante = pd.crosstab(
            df_filtrado[variable_x],
            df_filtrado[variable_y]
        )

        tabla_bivariante.loc["Total"] = tabla_bivariante.sum()


    elif medida == "Nota Media":

        tabla_bivariante = pd.pivot_table(
            df_filtrado,
            values="nota_final",
            index=variable_x,
            columns=variable_y,
            aggfunc="mean"
        ).round(2)
    
    elif medida == "Horas Estudio":

        tabla_bivariante = pd.pivot_table(
            df_filtrado,
            values="horas_estudio_semanal",
            index=variable_x,
            columns=variable_y,
            aggfunc="mean"
        ).round(2)


    else:

        tabla_bivariante = pd.pivot_table(
            df_filtrado,
            values="tasa_asistencia",
            index=variable_x,
            columns=variable_y,
            aggfunc="mean"
        ).round(2)


    st.dataframe(
        tabla_bivariante
    )


else:

    st.info(
    f"'Tasa de aprobados entre {variable_x} y {variable_y}"
    )   
    tabla_aprobacion = (
        pd.pivot_table(
            df_filtrado,
            values="aprobado_ml",
            index=variable_x,
            columns=variable_y,
            aggfunc="mean"
        ) * 100
    ).round(2)

    st.dataframe(
    tabla_aprobacion
    )
