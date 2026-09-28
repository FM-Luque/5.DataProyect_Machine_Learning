"""
sp_utils_ml.py
================
Cuaderno de referencia (bitácora) — igual que sp_utils.py y
sp_utils_visual.py, NO es un módulo con funciones para importar. Es
una chuleta de comandos sueltos de scikit-learn para clustering y
modelado, para no tener que buscarlos cada vez que uses tus propios
sp_clustering.py / sp_modelado.py del framework de Fase 3.

Cómo usarlo: busca la sección que necesites, copia el código y
adáptalo con tus columnas reales.

Todo está comentado con "#" a propósito, para que el archivo no dé
error si lo abres o ejecutas por accidente.
"""


# ============================================================================
# 0. SEPARAR VARIABLES — antes de cualquier modelo
# ============================================================================

# X = df.drop(columns=['objetivo_ml'])
# y = df['objetivo_ml']
#   X = todas las columnas que el modelo usará para PREDECIR (features).
#   y = la columna que el modelo tiene que ADIVINAR (target).
#   Convención: X siempre mayúscula, y siempre minúscula (así se llama
#   en toda la documentación de sklearn, ayuda a reconocer el patrón).

# X = df[['edad', 'ingresos', 'duracion']]
#   Si solo quieres usar unas pocas columnas como features, en vez de
#   "todas menos el objetivo", selecciónalas directamente así.


# ============================================================================
# 1. DIVIDIR EN ENTRENAMIENTO Y TEST
# ============================================================================

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#   Reparte los datos: 80% para entrenar el modelo (train), 20% para
#   comprobar si funciona con datos que NUNCA ha visto (test).
#   - test_size=0.2 -> 20% para test (cambia el nº si quieres otro reparto).
#   - random_state=42 -> fija la "semilla" aleatoria, para que el
#     reparto salga IGUAL cada vez que ejecutes el código (reproducible).

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
#   Igual, pero con stratify=y: mantiene la misma PROPORCIÓN de cada
#   categoría del objetivo en train y en test. Muy recomendable si tu
#   objetivo está desbalanceado (ej. mucho más 'no' que 'yes').


# ============================================================================
# 2. ESCALAR VARIABLES NUMÉRICAS
# ============================================================================

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train_esc = scaler.fit_transform(X_train)
# X_test_esc = scaler.transform(X_test)
#   Pone todas las columnas en la misma escala (media 0, desviación 1).
#   Necesario para modelos sensibles a la magnitud de los números
#   (KMeans, regresión logística, KNN...) — los basados en árboles
#   (Random Forest, árboles de decisión) NO lo necesitan.
#
#   OJO con el orden: fit_transform() SOLO en train (aprende la media
#   y desviación de ahí). transform() (sin fit) en test — así el test
#   se escala con los mismos parámetros del train, sin "hacer trampa"
#   mirando datos que se supone que no conoces todavía.


# ============================================================================
# 3. CLUSTERING — resumen rápido de tu propio flujo (sp_clustering.py)
# ============================================================================

# X_esc, scaler = sl.estandarizar_clustering(df, columnas=[...])
#   1. Escalar (ya tienes tu propia función para esto).

# mejor_k, score = sl.metodo_silhouette(X_esc, k_max=10)
#   2. Elegir el número de clusters (silhouette o método del codo).

# modelo, labels = sl.entrenar_kmeans(X_esc, n_clusters=mejor_k)
#   3. Entrenar KMeans con el k elegido.

# sl.evaluar_clustering(X_esc, labels, modelo)
#   4. Evaluar qué tan buenos son los clusters (silhouette, inercia...).

# sl.visualizar_clusters_pca(X_esc, labels)
#   5. Ver los clusters en un gráfico 2D (PCA reduce todas las
#   columnas a solo 2 dimensiones para poder dibujarlas).

# df['cluster'] = labels
# sl.perfil_clusters(df, labels)
#   6. Guardar a qué cluster pertenece cada fila, y ver el perfil medio
#   de cada grupo (para poder ponerles nombre: "clientes jóvenes con
#   pocos ingresos", etc.).


# ============================================================================
# 4. MODELOS DE CLASIFICACIÓN — predecir una categoría (ej. yes/no)
# ============================================================================

# from sklearn.linear_model import LogisticRegression

# modelo = LogisticRegression(max_iter=1000)
# modelo.fit(X_train_esc, y_train)
# predicciones = modelo.predict(X_test_esc)
#   El modelo de clasificación más simple y habitual como primera
#   prueba. fit() = entrena con lo que ya sabe (train). predict() =
#   adivina el resultado sobre datos que no ha visto (test).

# from sklearn.ensemble import RandomForestClassifier

# modelo = RandomForestClassifier(n_estimators=100, random_state=42)
# modelo.fit(X_train, y_train)
# predicciones = modelo.predict(X_test)
#   Otra alternativa habitual, normalmente da mejores resultados que
#   la regresión logística, y NO necesita las variables escaladas
#   (puedes pasarle X_train directo, sin el StandardScaler de arriba).


# ============================================================================
# 5. EVALUAR UN MODELO DE CLASIFICACIÓN
# ============================================================================

# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# accuracy_score(y_test, predicciones)
#   % de aciertos totales. Fácil de entender, pero engañoso si el
#   objetivo está desbalanceado (ej. 90% son 'no': un modelo que
#   siempre dice 'no' ya acertaría el 90% sin haber aprendido nada).

# print(classification_report(y_test, predicciones))
#   El resumen más completo: precision, recall y f1-score por cada
#   categoría. Mejor que accuracy solo cuando el objetivo está
#   desbalanceado (que es tu caso, con el dataset de bank-marketing).

# confusion_matrix(y_test, predicciones)
#   Tabla con aciertos y errores: cuántos 'yes' reales predijo bien
#   como 'yes', cuántos 'yes' reales confundió con 'no', etc.


# ============================================================================
# 6. MODELOS DE REGRESIÓN — predecir un número (no una categoría)
# ============================================================================

# from sklearn.linear_model import LinearRegression

# modelo = LinearRegression()
# modelo.fit(X_train, y_train)
# predicciones = modelo.predict(X_test)
#   Igual que LogisticRegression, pero para predecir un NÚMERO (ej.
#   ingresos, precio) en vez de una categoría.

# from sklearn.metrics import mean_squared_error, r2_score

# mean_squared_error(y_test, predicciones)
#   Error medio al cuadrado: cuanto más bajo, mejor. Como está "al
#   cuadrado", es sensible a errores grandes puntuales (outliers).

# r2_score(y_test, predicciones)
#   De 0 a 1 (puede salir negativo si el modelo es malo de verdad):
#   qué % de la variabilidad del dato explica el modelo. Más cercano
#   a 1 = mejor.


# ============================================================================
# 7. VALIDACIÓN CRUZADA — evaluar sin depender de un único reparto train/test
# ============================================================================

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(modelo, X, y, cv=5)
# scores.mean()
#   En vez de un único train/test, divide los datos en 5 partes (cv=5),
#   entrena y evalúa 5 veces (cada vez con una parte distinta como
#   test), y te da los 5 resultados. Más fiable que un solo
#   train_test_split(), porque no depende de "haber tenido suerte" con
#   un reparto concreto.


# ============================================================================
# 8. GUARDAR Y CARGAR UN MODELO YA ENTRENADO
# ============================================================================

# import joblib

# joblib.dump(modelo, 'modelo_final.pkl')
#   Guarda el modelo entrenado en un archivo, para no tener que
#   reentrenarlo cada vez que abras el notebook.

# modelo_cargado = joblib.load('modelo_final.pkl')
#   Recupera el modelo guardado, listo para usar con .predict()
#   directamente, sin pasar otra vez por fit().


# ============================================================================
# 9. QUÉ VARIABLES PESAN MÁS EN EL MODELO
# ============================================================================

# importancias = pd.Series(modelo.feature_importances_, index=X.columns)
# importancias.sort_values(ascending=False)
#   Solo para modelos de árboles (Random Forest, árboles de decisión):
#   qué columnas influyen más en la predicción. Útil para explicar el
#   modelo o para descartar variables que no aportan nada.

# coeficientes = pd.Series(modelo.coef_[0], index=X.columns)
# coeficientes.sort_values(ascending=False)
#   El equivalente para LogisticRegression/LinearRegression: el signo
#   indica si la variable empuja hacia 'sí' (+) o hacia 'no' (-), y el
#   tamaño indica cuánto peso tiene (con las variables ya escaladas,
#   si no, los tamaños no son comparables entre columnas).
