"""
sp_modelado.py

Caja de herramientas de Regresión & Clasificación (Machine Learning).

FLUJO RECOMENDADO
0. Importar librerias
1. Cargar datos + copy()
2. Analisis inicial
    clasificar_columnas()
3. Ppreparacion de datos
    preparacion_dataframe()
4. Separar variables
    separar_xy()
5. Transformar variables
    tratar_numericas()
    codificar_binarias()
    codificar_categoricas()
6. Division Train / Test
    train_test()
7. Estandarizacion
    estandarizar()
8. Entrenamiento inicial
    entrenar_evaluar_clasificacion()
    entrenar_evaluar_regresion()
9. Metricas
    metricas_clasificacion()
    metricas_regresion()
10. Validación
    comparar_train_test()
11. Comparacion de grupos
    comparar_modelos()
12. Optimizacion
    grid_search()
13. Entrenamiento final
    entrenar_final()
14. Interpretacion
    importancia_variables()
15. Simulacion
    simular_variable()
16. Predicciones
    predecir()
17. Guardar modelo 
    guardar_modelo()
18. Cargar modelo
    cargar_modelo()

Nota sobre el orden: los números de sección son categorías de
función, no pasos estrictamente secuenciales. En particular,
comparar_train_test() (sección 2) se usa DESPUÉS de entrenar el
modelo (sección 3), ya que necesita un modelo ya entrenado para
comparar sus predicciones en train y test.

Los MODELOS (LinearRegression, LogisticRegression, Ridge, Lasso,
ElasticNet, DecisionTree, RandomForest, XGBoost...) se crean
directamente con sklearn, o usa comparar_modelos() para probar
varios a la vez con sus valores por defecto.

Este módulo encapsula únicamente las tareas repetitivas.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import (
    StandardScaler
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,

    accuracy_score,
    precision_score,
    recall_score,
    f1_score,

    confusion_matrix,
    classification_report
)

from sklearn.linear_model import (
    LinearRegression, 
    LogisticRegression, 
    Ridge, 
    Lasso, 
    ElasticNet)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Para que se muestren todas las columnas al inspeccionar los DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

"""
# ==========================================================
# MODELOS DE REGRESIÓN
# TV numérica
# ==========================================================

modelo_linear = LinearRegression()

modelo_ridge = Ridge(
    alpha=1.0
)

modelo_lasso = Lasso(
    alpha=0.1
)

modelo_elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

# ==========================================================
# MODELOS DE CLASIFICACIÓN
# TV binaria o categórica
# ==========================================================

modelo_logistic = LogisticRegression(
    max_iter=1000
)

modelo_tree = DecisionTreeClassifier(
    random_state=42
)

modelo_forest = RandomForestClassifier(
    random_state=42
)
"""

# ============================================================================
# 2. ANALISIS INICIAL
# ============================================================================

def clasificar_columnas(df, max_categorias_control=15, min_unicos_metrica=15):
    """
    Analiza cada columna del DataFrame y sugiere si es candidata a
    'col_control' (variable de agrupación categórica) o a 'metrica'
    (variable numérica a medir), útil como paso previo a normalidad(),
    homocedasticidad() y los tests de sp_abtest.py.

    Parameters
    ----------
    df : DataFrame
    max_categorias_control : int
        Nº máximo de valores únicos para considerar una columna categórica
        como candidata razonable a col_control (por defecto 15).
    min_unicos_metrica : int
        Nº mínimo de valores únicos para considerar una columna numérica
        como candidata CLARA a metrica sin revisar (por defecto 15).
        Por debajo de este umbral, se marca "revisar" en vez de descartar,
        porque puede ser una escala válida (ej. calificacion 1-5).

    Returns
    -------
    DataFrame con columnas: columna, dtype, n_unicos, sugerencia, motivo
    """
    pd.set_option('display.max_colwidth', None)
    filas = []

    for col in df.columns:
        dtype = df[col].dtype
        n_unicos = df[col].nunique()
        es_numerica = pd.api.types.is_numeric_dtype(dtype)
        es_fecha = pd.api.types.is_datetime64_any_dtype(dtype)

        if es_fecha:
            sugerencia = "descartada"
            motivo = "Es fecha; usar columnas derivadas (año, mes) como col_control si aplica"

        elif es_numerica:
            if n_unicos >= min_unicos_metrica:
                sugerencia = "metrica"
                motivo = f"Numérica con {n_unicos} valores únicos: variable continua medible"
            elif n_unicos <= 1:
                sugerencia = "descartada"
                motivo = "Solo 1 valor único: no aporta información"
            else:
                sugerencia = "revisar"
                motivo = f"Numérica con solo {n_unicos} valores únicos: puede ser escala válida (ej. calificación 1-5) o código mal tipado — revisar a mano"

        else:
            if 2 <= n_unicos <= max_categorias_control:
                sugerencia = "col_control"
                motivo = f"Categórica con {n_unicos} valores únicos: rango razonable para agrupar"
            elif n_unicos == 1:
                sugerencia = "descartada"
                motivo = "Solo 1 valor único: no permite comparar grupos"
            else:
                sugerencia = "descartada"
                motivo = f"Categórica con {n_unicos} valores únicos: demasiados para agrupar de forma útil (ej. ID, nombre)"

        filas.append({
            "columna": col,
            "dtype": str(dtype),
            "n_unicos": n_unicos,
            "sugerencia": sugerencia,
            "motivo": motivo
        })

    resultado = pd.DataFrame(filas)
    orden = {"col_control": 0, "metrica": 1, "revisar": 2, "descartada": 3}
    resultado["orden_tmp"] = resultado["sugerencia"].map(orden)
    resultado = resultado.sort_values(["orden_tmp", "n_unicos"]).drop(columns="orden_tmp").reset_index(drop=True)

    return resultado



# ============================================================================
# 3. PREPARACION DE DATOS 
# ============================================================================

def preparacion_dataframe(
    df,
    eliminar_duplicados=True,
    reset_index=True
):
    """
    Preparación básica del DataFrame antes del modelado.

    - Elimina duplicados.
    - Reinicia índices.
    - Devuelve una copia limpia.

    Esta función NO modifica el DataFrame original.
    """

    df = df.copy()

    if eliminar_duplicados:
        df = df.drop_duplicates()

    if reset_index:
        df = df.reset_index(drop=True)

    return df

# ============================================================================
# 4. SEPARAR VARIABLES
# ============================================================================

def separar_xy(df, objetivo):
    """
    Separa DataFrame en X (predictoras)
    e y (objetivo).
    """

    X = df.drop(columns=[objetivo])
    y = df[objetivo]

    return X, y


# ============================================================================
# 5. TRANSFORMAR VARIABLES
# ============================================================================

def tratar_numericas(
    df,
    columnas=None
):
    """
    Verifica y convierte variables numéricas.

    Se asume que el tratamiento de nulos
    ya ha sido realizado durante la fase
    de limpieza de datos.
    """

    df = df.copy()

    if columnas is None:

        columnas = df.select_dtypes(
            include=np.number
        ).columns

    for col in columnas:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


def tratar_categoricas(
    df,
    columnas
):
    """
    Limpieza básica de variables categóricas.

    - Convierte a str.
    - Elimina espacios al inicio y final.

    Recomendación:
    Ejecutar antes de codificar_binarias()
    y codificar_categoricas().
    """

    df = df.copy()

    for col in columnas:

        df[col] = (
            df[col]
            .astype("str")
            .str.strip()
        )

    return df


def codificar_binarias(
    df,
    columnas,
    mapas=None
):
    """
    Convierte variables binarias texto
    a formato 0/1.

Ejemplo de uso
    mapas = {"sexo": {
            "masculino": 0,
            "femenino": 1},
        "cabina_info": {
            "Desconocida": 0,
            "Conocida": 1} 
            }

    """

    df = df.copy()

    for col in columnas:

        valores = sorted(
            df[col].dropna().unique()
        )

        if len(valores) != 2:
            raise ValueError(
                f"{col} no es binaria."
            )

        if mapas and col in mapas:

            df[col] = df[col].map(
                mapas[col]
            )

        else:

            df[col] = df[col].map({
                valores[0]: 0,
                valores[1]: 1
            })

    return df


def codificar_categoricas(
    df,
    columnas,
):
    """
    One-Hot Encoding.

    Genera todas las categorías disponibles
    (drop_first=False).

    Modelado recomendado:
    ---------------------
    # Xc con todas las referencias
    
    Xc = sm.codificar_categoricas(X,columnas)

    # Xs sin primeras referencias
    Xs = Xc.drop(columns=[" "," "], errors="ignore")

    A continuación se pueden comparar ambas versiones mediante:

        train_test()
        estandarizar()
        comparar_modelos()

    Si no existe mejora relevante, se recomienda mantener Xc.
    """

    df = pd.get_dummies(
        df,
        columns=columnas,
        drop_first=False
    )

    columnas_bool = df.select_dtypes(
        include=["bool", "boolean"]
    ).columns

    df[columnas_bool] = (
        df[columnas_bool]
        .astype(int)
    )

    return df



# ============================================================================
# 6. DIVIDIR TRAIN/TEST
# ============================================================================
def train_test(
    X,
    y,
    test_size=0.2,
    random_state=42,
    estratificar=False
):
    """
    División train/test.
    
    Clasificación → estratificar=True
    Regresión → estratificar=False
    
    """

    stratify = y if estratificar else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

# ============================================================================
# 7. ESTANDARIZAR VARIABLES
# ============================================================================
def estandarizar(
    X_train,
    X_test,
    columnas=None
):
    """
    Estandarización mediante StandardScaler.
    """

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    if columnas is None:

        columnas = X_train.select_dtypes(
            include=np.number
        ).columns

    X_train[columnas] = scaler.fit_transform(
        X_train[columnas]
    )

    X_test[columnas] = scaler.transform(
        X_test[columnas]
    )

    return X_train, X_test, scaler


# ============================================================================
# 8. ENTRENAMIENTO INICIAL
# ============================================================================

def entrenar_evaluar_clasificacion(
    modelo,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Entrena y evalúa un modelo de clasificación.

    Ejemplo uso 
    modelo = LogisticRegression(
    max_iter=1000,
    random_state=42
    )
 
    modelo, metricas, y_pred = (sm.entrenar_evaluar_clasificacion(modelo,X_train,X_test,y_train,y_test))

    """
    modelo.fit(
        X_train,
        y_train
    )

    y_pred = modelo.predict(
        X_test
    )

    print("\nMatriz de confusión:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    metricas = metricas_clasificacion(
        y_test,
        y_pred
    )

    return (modelo, metricas,y_pred) 

def entrenar_evaluar_regresion(
    modelo,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Entrena y evalúa un modelo de regresión.

    Ejemplo uso 
        modelo = 
     
        modelo, metricas, y_pred = (sm.entrenar_evaluar_regresion(modelo,X_train,X_test,y_train,y_test))

    """

    modelo.fit(
        X_train,
        y_train
    )

    y_pred = modelo.predict(
        X_test
    )

    metricas = metricas_regresion(
        y_test,
        y_pred
    )

    return (
        modelo,
        metricas,
        y_pred
    )

""" 
    Antes de entrenar_evaluar_regresion:

    modelo = LogisticRegression(
    max_iter=1000,
    random_state=42)
    """


# ============================================================================
# 9. MÉTRICAS
# ============================================================================


def metricas_regresion(
    y_real,
    y_pred
):
    """
    Métricas para regresión.
    """

    mse = mean_squared_error(
        y_real,
        y_pred
    )

    resultados = {
        "MAE": mean_absolute_error(
            y_real,
            y_pred
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(
            y_real,
            y_pred
        )
    }

    return resultados


def metricas_clasificacion(
    y_real,
    y_pred
):
    """
    Métricas para clasificación.
    """

    resultados = {

        "Accuracy": accuracy_score(
            y_real,
            y_pred
        ),

        "Precision": precision_score(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Recall": recall_score(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "F1": f1_score(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


    return resultados


# ============================================================================
# 10. VALIDACION 
# ============================================================================


def comparar_train_test(
    modelo,
    X_train,
    X_test,
    y_train,
    y_test,
    tipo="regresion"
):
    """
    Compara métricas train/test para
    detectar overfitting o underfitting.
    comparar_train_test() (sección 2) se usa DESPUÉS de entrenar el
    modelo (sección 3), ya que necesita un modelo ya entrenado para
    comparar sus predicciones en train y test.
    """

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    if tipo == "regresion":

        r2_train = r2_score(
            y_train,
            pred_train
        )

        r2_test = r2_score(
            y_test,
            pred_test
        )

        print(f"R² Train: {r2_train:.4f}")
        print(f"R² Test : {r2_test:.4f}")

    elif tipo == "clasificacion":

        acc_train = accuracy_score(
            y_train,
            pred_train
        )

        acc_test = accuracy_score(
            y_test,
            pred_test
        )

        print(f"Accuracy Train: {acc_train:.4f}")
        print(f"Accuracy Test : {acc_test:.4f}")

    else:

        raise ValueError(
            "tipo debe ser "
            "'regresion' o "
            "'clasificacion'"
        )


# ============================================================================
# 11. COMPARACION DE MODELOS 
# ============================================================================
def comparar_modelos(X_train, X_test, y_train, y_test, tipo='regresion', modelos=None):
    """
    Entrena y compara varios modelos a la vez sobre el mismo train/test,
    para decidir cuál usar antes de afinarlo con grid_search().

    Si no se indica 'modelos', usa un conjunto por defecto según tipo:
    - regresion: Linear, Ridge, Lasso, ElasticNet
    - clasificacion: Logistic

    Parameters
    ----------
    X_train, X_test, y_train, y_test : conjuntos ya preparados.
    tipo : str
        'regresion' o 'clasificacion'.
    modelos : dict, opcional
        {nombre: instancia_del_modelo}, ej.
        {'Linear': LinearRegression(), 'Ridge': Ridge(alpha=1.0)}.
        Si no se indica, se usa el conjunto por defecto de arriba.

    Returns
    -------
    DataFrame con las métricas de cada modelo, una fila por modelo.

    Ejemplo de uso
    -------
    sm.comparar_modelos(X_train_reg, X_test_reg, y_train_reg, y_test_reg, tipo='regresion')  
    """
    if modelos is None:
        if tipo == 'regresion':
            modelos = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
            }
        elif tipo == 'clasificacion':
            modelos = {"Logistic": LogisticRegression(max_iter=1000),                       
                       "Tree": DecisionTreeClassifier(random_state=42),
                       "Forest": RandomForestClassifier(random_state=42)}
        else:
            raise ValueError('tipo debe ser "regresion" o "clasificacion"')

    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        pred_train = modelo.predict(X_train)
        pred_test = modelo.predict(X_test)

        if tipo == 'regresion':
            resultados[nombre] = {
                'R2_train': r2_score(y_train, pred_train),
                'R2_test': r2_score(y_test, pred_test),
                'MAE_test': mean_absolute_error(y_test, pred_test),
                'RMSE_test': np.sqrt(mean_squared_error(y_test, pred_test)),
            }
        elif tipo == 'clasificacion':
            resultados[nombre] = {
                'Accuracy_train': accuracy_score(y_train, pred_train),
                'Accuracy_test': accuracy_score(y_test, pred_test),
                'F1_test': f1_score(y_test, pred_test, average='weighted', zero_division=0),
            }

    return pd.DataFrame(resultados).T.round(4)



# ============================================================================
# 12. OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================================

def grid_search(
    modelo,
    param_grid,
    X_train,
    y_train,
    cv=5,
    scoring=None
):
    """
    Optimiza automáticamente los hiperparámetros de un modelo
    mediante GridSearchCV.

    Prueba todas las combinaciones definidas en param_grid y
    selecciona la que obtiene mejor resultado mediante validación
    cruzada.

    Parameters
    ----------
    modelo : estimador sklearn
        Modelo que se desea optimizar.

    param_grid : dict
        Diccionario con los hiperparámetros a probar.

        Ejemplos habituales:

        LogisticRegression:
            {
                "C": [0.01, 0.1, 1, 10, 100]
            }

        DecisionTreeClassifier:
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }

        RandomForestClassifier:
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None]
            }

        Ridge:
            {
                "alpha": [0.01, 0.1, 1, 10, 100]
            }

        Lasso:
            {
                "alpha": [0.001, 0.01, 0.1, 1]
            }

        ElasticNet:
            {
                "alpha": [0.01, 0.1, 1],
                "l1_ratio": [0.2, 0.5, 0.8]
            }

    X_train : DataFrame
        Variables predictoras de entrenamiento.

    y_train : Series
        Variable objetivo de entrenamiento.

    cv : int, default=5
        Número de particiones utilizadas en validación cruzada.

    scoring : str, opcional
        Métrica de evaluación.

        Ejemplos:
            Clasificación:
                "accuracy"
                "f1"
                "precision"
                "recall"

            Regresión:
                "r2"
                "neg_mean_squared_error"
                "neg_mean_absolute_error"

    Returns
    -------
    sklearn estimator

        Devuelve el modelo optimizado con los mejores parámetros.

    Ejemplo
    --------

    LogisticRegression:

        modelo = LogisticRegression(max_iter=1000)

        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100]
        }

        mejor_modelo = sm.grid_search(
            modelo,
            param_grid,
            X_train,
            y_train,
            scoring="accuracy"
        )

    Notes
    -----
    comparar_modelos()
        ↓
    Selecciona el mejor modelo

    grid_search()
        ↓
    Optimiza sus hiperparámetros
    """

    grid = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    grid.fit(
        X_train,
        y_train
    )

    print("\nMejores parámetros:")
    print(grid.best_params_)

    return grid.best_estimator_


# ============================================================================
# 13. ENTRENAMIENTO FINAL
# ============================================================================

def entrenar_final(modelo, X, y):
    """
    Reentrena el modelo con el 100% de los datos (X, y completos, sin
    dividir en train/test), una vez ya validado su rendimiento con
    train_test(). Úsalo justo antes de guardar_modelo(), para
    aprovechar todo el dato disponible en el modelo de producción.

    Parameters
    ----------
    modelo : instancia de sklearn ya elegida (sin entrenar, o se
        reentrena igualmente sobre el 100% del dato).
    X, y : dataset COMPLETO (no X_train/y_train).

    Returns
    -------
    El modelo, ya entrenado con todos los datos.
    """
    modelo.fit(X, y)
    return modelo

# ============================================================================
# 14. INTERPRETACIÓN DEL MODELO
# ============================================================================

def importancia_variables(
    modelo,
    columnas,
    top_n=15
):
    """
    Soporta:

    - coef_ (modelos lineales: LinearRegression, LogisticRegression,
      Ridge, Lasso...) -> se ordena por magnitud absoluta, pero se
      muestra el coeficiente CON SU SIGNO, para poder distinguir si
      la relación es positiva (a más X, más y) o negativa (a más X,
      menos y). Perder el signo perdería justo esa información.

    - feature_importances_ (árboles: DecisionTree, RandomForest,
      GradientBoosting...) -> siempre positivo, no tiene signo
      (no indica dirección, solo cuánto pesa la variable).
    """

    if hasattr(modelo, "coef_"):

        valores = np.ravel(modelo.coef_)  # con signo, sin abs()

    elif hasattr(
        modelo,
        "feature_importances_"
    ):

        valores = modelo.feature_importances_

    else:

        raise ValueError(
            "Modelo sin coef_ ni feature_importances_"
        )

    imp = pd.DataFrame({
        "variable": columnas,
        "importancia": valores
    })

    imp["abs"] = imp["importancia"].abs()

    imp = imp.sort_values(
        by="abs",
        ascending=False
    ).drop(columns="abs").head(top_n)

    colores = ["#d62728" if v < 0 else "#1f77b4" for v in imp["importancia"]]

    plt.figure(figsize=(8, 5))

    plt.barh(
        imp["variable"],
        imp["importancia"],
        color=colores
    )

    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", linewidth=0.8)

    plt.title(
        "Importancia de Variables"
    )

    plt.tight_layout()

    plt.show()

    return imp


# ============================================================================
# 15. SIMULACION DE ESCENARIOS
# ============================================================================
def simular_variable(
    modelo,
    X,
    columna,
    valores,
    tipo="auto"
):
    """
    Simula cómo cambia la predicción del modelo al modificar UNA
    variable y mantener el resto fijas en su valor medio.
     
    Es una herramienta de interpretación que permite observar el
    comportamiento del modelo más allá de la importancia de variables
    o los coeficientes.
     
    Parameters
    ----------
    modelo : modelo entrenado
    Modelo de sklearn previamente ajustado mediante .fit().
     
    X : DataFrame
    Dataset utilizado para entrenar el modelo. Se emplea para
    construir un perfil medio de referencia.
     
    columna : str
    Variable que se desea modificar.
     
    valores : list o array
    Valores que tomará la variable simulada.
     
    tipo : str, default="auto"
    Tipo de predicción a devolver.
     
    - "auto":
    Detecta automáticamente si el modelo dispone de
    predict_proba() y devuelve probabilidades.
    - "clasificacion":
    Devuelve la probabilidad estimada de la clase positiva.
    - "regresion":
    Devuelve la predicción numérica del modelo.
     
    Returns
    -------
    DataFrame
     
    Clasificación:
    Contiene la variable simulada y la probabilidad estimada
    de pertenecer a la clase positiva.
     
    Ejemplo:
    clase probabilidad
    -1.5 0.49
    -0.3 0.37
    0.8 0.27
     
    Regresión:
    Contiene la variable simulada y la predicción del modelo.
     
    Ejemplo:
    edad prediccion
    20 125000
    40 182000
    60 210000
     
    Notes
    -----
    En modelos de clasificación suele ser más útil analizar
    probabilidades mediante predict_proba() que clases finales
    (0 o 1), ya que permite observar cambios graduales en el
    comportamiento del modelo.
    """

    perfil_base = X.mean()

    filas = []

    for v in valores:
        fila = perfil_base.copy()
        fila[columna] = v
        filas.append(fila)

    X_simulado = pd.DataFrame(
        filas
    )[X.columns]

    # Clasificación
    if (
        tipo == "clasificacion"
        or (
            tipo == "auto"
            and hasattr(
                modelo,
                "predict_proba"
            )
        )
    ):

        predicciones = modelo.predict_proba(
            X_simulado
        )[:, 1]

        return pd.DataFrame({
            columna: valores,
            "probabilidad": predicciones.round(4)
        })

    # Regresión
    predicciones = modelo.predict(
        X_simulado
    )

    return pd.DataFrame({
        columna: valores,
        "prediccion": np.round(
            predicciones,
            2
        )
    })



# ============================================================================
# 16. PREDICCIONES
# ============================================================================


def predecir(
    modelo,
    X,
    probabilidades=False
):
    """
    Realiza predicciones con un modelo
    previamente entrenado.
    """
    if probabilidades:
        return modelo.predict_proba(X)

    return modelo.predict(X)



# ============================================================================
# 17. GUARDAR MODELO
# ============================================================================

def guardar_modelo(
    modelo,
    ruta
):
    """
    Guarda un modelo en formato .pkl
   
    ruta = '../data/processed/nombre_modelo_guardar.pkl'
    """
    joblib.dump(
        modelo,
        ruta
    )

    print(
        f"Modelo guardado en: {ruta}"
    )
""" 
Comprobacion de carga 
import os
 
os.path.exists(
"modelo_titanic.pkl"
)
"""

# ============================================================================
# 18. CARGAR MODELO
# ============================================================================


"""
    Carga un modelo .pkl
    
    Ejemplo de uso típico, al principio de un notebook NUEVO de
    predicción (no de entrenamiento):

        modelo = sm.cargar_modelo('modelo_regresion_final.pkl')
        predicciones = sm.predecir(modelo, datos_nuevos)

    Parameters
    ----------
    ruta : str
        ej. '../data/processed/nombre_modelo.pkl'

    """



def cargar_modelo(
    ruta
):
    """
    Carga un modelo guardado.
    """

    modelo = joblib.load(
        ruta
    )

    print(
        f"Modelo cargado desde: {ruta}"
    )

    return modelo