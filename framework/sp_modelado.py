"""
sp_modelado.py

Caja de herramientas de Regresión & Clasificación (Machine Learning).

Sigue el flujo del temario:

1. PREPARAR DATOS
    separar_xy()
    train_test()
    codificar_categoricas()
    estandarizar()

2. MÉTRICAS
    metricas_regresion()
    metricas_clasificacion()
    comparar_train_test()

3. ENTRENAR Y EVALUAR
    entrenar_evaluar_regresion()
    entrenar_evaluar_clasificacion()
    predecir()

4. OPTIMIZACIÓN
    grid_search()
    comparar_modelos()

5. INTERPRETACIÓN
    importancia_variables()
    simular_variable( )

6. PRODUCCIÓN
    entrenar_final()
    guardar_modelo()
    cargar_modelo()

Los MODELOS (LinearRegression, LogisticRegression,
DecisionTree, RandomForest, XGBoost...) se crean
directamente con sklearn.

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


# Para que se muestren todas las columnas al inspeccionar los DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# ============================================================================
# 1. PREPARACIÓN
# ============================================================================

def separar_xy(df, objetivo):
    """
    Separa DataFrame en X (predictoras)
    e y (objetivo).
    """

    X = df.drop(columns=[objetivo])
    y = df[objetivo]

    return X, y


def train_test(
    X,
    y,
    test_size=0.2,
    random_state=42,
    estratificar=False
):
    """
    División train/test.
    """

    stratify = y if estratificar else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )


def codificar_categoricas(
    df,
    columnas,
    eliminar_primera=True
):
    """
    One-Hot Encoding.
    """

    return pd.get_dummies(
        df,
        columns=columnas,
        drop_first=eliminar_primera
    )


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
# 2. MÉTRICAS
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

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_real, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_real,
            y_pred,
            zero_division=0
        )
    )

    return resultados


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
# 3. ENTRENAMIENTO Y EVALUACIÓN
# ============================================================================

def entrenar_evaluar_regresion(
    modelo,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Entrena y evalúa un modelo de regresión.
    """

    modelo.fit(
        X_train,
        y_train
    )

    pred = modelo.predict(X_test)

    return metricas_regresion(
        y_test,
        pred
    )


def entrenar_evaluar_clasificacion(
    modelo,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Entrena y evalúa un modelo de clasificación.
    """

    modelo.fit(
        X_train,
        y_train
    )

    pred = modelo.predict(X_test)

    return metricas_clasificacion(
        y_test,
        pred
    )


def predecir(
    modelo,
    X
):
    """
    Realiza predicciones con un modelo
    previamente entrenado.
    """

    return modelo.predict(X)


# ============================================================================
# 4. OPTIMIZACIÓN
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
    Optimización mediante GridSearchCV.
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


def comparar_modelos(modelos, X_train, X_test, y_train, y_test, tipo='regresion'):
    """
    Entrena y compara varios modelos a la vez sobre el mismo train/test,
    para decidir cuál usar antes de afinarlo con grid_search().

    Parameters
    ----------
    modelos : dict
        Diccionario {nombre: instancia_del_modelo}, ej.
        modelos = {'Linear': LinearRegression(), 'Ridge': Ridge(alpha=1.0), 'Lasso': Lasso(alpha=0.1),"ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)}
    X_train, X_test, y_train, y_test : conjuntos ya preparados.
    tipo : str
        'regresion' o 'clasificacion'.

    Returns
    -------
    DataFrame con las métricas de cada modelo, una fila por modelo.
    """
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
        else:
            raise ValueError('tipo debe ser "regresion" o "clasificacion"')

    return pd.DataFrame(resultados).T.round(4)



# ============================================================================
# 5. INTERPRETACIÓN
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

def simular_variable(modelo, X, columna, valores):
    """
    Simula cómo cambia la predicción del modelo al variar UNA columna,
    dejando el resto fijas en su valor medio.
    """
    X_simulado = pd.DataFrame([X.mean()] * len(valores))
    X_simulado[columna] = valores
    predicciones = modelo.predict(X_simulado)
    return pd.DataFrame({columna: valores, 'prediccion': predicciones.round(2)})

# ============================================================================
# 6. PRODUCCIÓN
# ============================================================================
# 

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
    
    X = X_reg_completo = pd.concat([X_train_reg, X_test_reg])
    y =y_reg_completo = pd.concat([y_train_reg, y_test_reg])
    
    Returns
    -------
    El modelo, ya entrenado con todos los datos.
    """
    modelo.fit(X, y)
    return modelo


def guardar_modelo(
    modelo,
    ruta
):
    """
    Guarda un modelo en formato .pkl
    05_regresion.ipynb   -> entrenas, evalúas, guardar_modelo()  (aquí SÍ va al final)

    ruta = '../data/processed/nombre_modelo_guardar.pkl'
    """

    joblib.dump(
        modelo,
        ruta
    )


def cargar_modelo(
    ruta
):
    """
    Carga un modelo .pkl
    ruta = '../data/processed/nombre_modelo_cargar.pkl'

    
    """

    return joblib.load(ruta)

    """
    08_prediccion.ipynb (un notebook NUEVO, quizás semanas después)
   -> primera línea: modelo = sm.cargar_modelo('modelo_regresion_final.pkl')
   -> segunda línea: predicciones = sm.predecir(modelo, datos_nuevos)
    """