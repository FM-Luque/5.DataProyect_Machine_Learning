# 5.DataProyect_Machine_Learning - Rendimiento Académico de Estudiantes

# 🎓 Proyecto Machine Learning - Rendimiento Académico de Estudiantes

## 📖 Descripción

Proyecto completo de Machine Learning desarrollado durante el módulo de Regresión y Clasificación.

El objetivo consiste en analizar los factores relacionados con el rendimiento académico de estudiantes mediante técnicas de análisis exploratorio de datos (EDA), preprocesamiento, regresión y clasificación.

Se han desarrollado dos modelos:

- Regresión para predecir la nota final del estudiante.
- Clasificación para predecir si un estudiante aprobará o suspenderá.

---

## 🎯 Objetivos

- Realizar un análisis exploratorio completo del dataset.
- Limpiar y preparar los datos para Machine Learning.
- Construir un modelo de regresión para predecir la nota final.
- Construir un modelo de clasificación para predecir la aprobación.
- Identificar las variables con mayor influencia en el rendimiento académico.

---

## 🗂️ Estructura del Proyecto

```text
5.DataProyect_Machine_Learning/

├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── docs/
│
├── framework/
│   ├── sp_carga_exploracion.py
│   ├── sp_limpieza_transformacion.py
│   ├── sp_eda.py
│   └── sp_modelado.py
│
├── notebooks/
│   ├── 01_carga_exploracion.ipynb
│   ├── 02_limpieza_transformacion.ipynb
│   ├── 03_eda.ipynb
│   ├── 05_preprocesamiento.ipynb
│   ├── 06_regresion.ipynb
│   └── 07_clasificacion.ipynb
│
├── src/
│   ├── sp_utils.py
│   ├── sp_utils_ml.py
│   └── sp_utils_visual.py
│
├── informe/
│
├── README.md
└── requirements.txt
```

---

## 📊 Dataset

El dataset contiene información académica de 1000 estudiantes.

### Variables principales

- horas_estudio_semanal
- nota_anterior
- tasa_asistencia
- horas_sueno
- edad
- nivel_dificultad
- tiene_tutor
- horario_estudio_preferido
- estilo_aprendizaje

### Variables objetivo

**Regresión**

- nota_final

**Clasificación**

- aprobado_ml

---

## 🧹 Preprocesamiento realizado

- Tratamiento de valores nulos.
- Imputación de variables numéricas mediante media.
- Imputación de variables categóricas mediante "desconocido".
- Estandarización de texto.
- Eliminación de tildes.
- Conversión de variables binarias.
- One Hot Encoding.
- Validación de reglas de negocio.
- Detección y revisión de outliers.

---

## 📈 Resultados EDA

### Factores más relacionados con el rendimiento

- Horas de estudio semanal.
- Nota anterior.
- Disponibilidad de tutor.
- Tasa de asistencia.

### Factores con poca influencia

- Edad.
- Horas de sueño.
- Estilo de aprendizaje.
- Horario de estudio preferido.

### Hallazgo destacado

Los estudiantes con tutor presentan:

- Más horas de estudio.
- Mejor nota final.
- Mayor tasa de aprobación.

---

## 🤖 Modelo de Regresión

### Modelo seleccionado

Lasso Regression

### Resultados

| Métrica | Valor |
| -------- | ----- |
| R²      | 0.368 |
| MAE      | 5.84  |
| RMSE     | 7.19  |

### Variables más influyentes

- Nivel de dificultad.
- Horas de estudio.
- Tutor.
- Nota anterior.

---

## 🤖 Modelo de Clasificación

### Modelo recomendado

Logistic Regression

### Resultados

| Métrica | Valor |
| -------- | ----- |
| Accuracy | 0.92  |
| F1 Score | 0.90  |

### Consideración importante

El dataset presenta desbalanceo:

- 89.8% Aprobados
- 10.2% Suspensos

Por ello el Accuracy debe interpretarse junto con Precision, Recall y F1.

---

## 📌 Conclusiones

Los factores con mayor capacidad predictiva del rendimiento académico son:

1. Horas de estudio semanal.
2. Nota anterior.
3. Disponibilidad de tutor.
4. Tasa de asistencia.

La edad, las horas de sueño y el estilo de aprendizaje muestran una influencia limitada sobre los resultados académicos.

---


## 👨‍💻 Autor

Proyecto desarrollado por Paco Luque como práctica final del módulo Regresión y Clasificación.
