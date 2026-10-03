
# Informe Final Proyecto Machine Learning

## 1. Introducción

El presente proyecto tiene como objetivo analizar el rendimiento académico de estudiantes mediante técnicas de análisis exploratorio de datos y Machine Learning.

Se desarrollan dos problemas:

- Regresión: predicción de la nota final.
- Clasificación: predicción de aprobación o suspensión.

---

## 2. Metodología

El proyecto se ha dividido en seis fases:

1. Carga y exploración inicial.
2. Limpieza y transformación.
3. Análisis exploratorio de datos.
4. Preprocesamiento para Machine Learning.
5. Modelado de regresión.
6. Modelado de clasificación.

---

## 3. Calidad de los datos

Dataset inicial:

- 1000 registros.
- 11 variables.

Problemas detectados:

- Valores nulos.
- Inconsistencias de formato de texto.

Acciones realizadas:

- Imputación de nulos.
- Normalización de categorías.
- Validación de reglas de negocio.
- Revisión de outliers.

Resultado:

- Dataset sin nulos.
- Dataset sin duplicados.
- Dataset validado.

---

## 4. Resultados del EDA

### Factores más relevantes

- Horas de estudio semanal.
- Nota anterior.
- Presencia de tutor.
- Tasa de asistencia.

### Factores menos relevantes

- Edad.
- Horas de sueño.
- Estilo de aprendizaje.
- Horario de estudio.

### Hallazgo principal

Los estudiantes con tutor muestran:

- Más horas de estudio.
- Mayor tasa de aprobación.
- Mejor rendimiento académico global.

---

## 5. Modelo de Regresión

### Objetivo

Predecir la nota final.

### Modelo seleccionado

Lasso Regression.

### Resultados

- R² = 0.368
- MAE = 5.84
- RMSE = 7.19

### Conclusión

Las variables académicas explican parcialmente el rendimiento final del estudiante.

---

## 6. Modelo de Clasificación

### Objetivo

Predecir si el alumno aprobará.

### Modelo seleccionado

Logistic Regression.

### Resultados

- Accuracy = 92%
- F1 Score = 0.90

### Limitación

El conjunto de datos presenta un fuerte desbalanceo entre aprobados y suspensos.

---

## 7. Conclusiones finales

Los factores más relacionados con el éxito académico son:

- Horas de estudio.
- Nota anterior.
- Presencia de tutor.
- Asistencia.

Los modelos obtenidos permiten identificar patrones útiles para comprender el rendimiento académico y constituyen una base sólida para desarrollos posteriores.

---

## 8. Próximos pasos

- Aplicar balanceo de clases.
- Incorporar nuevos algoritmos.
- Desarrollar dashboard interactivo.
- Desplegar modelos para inferencia.
