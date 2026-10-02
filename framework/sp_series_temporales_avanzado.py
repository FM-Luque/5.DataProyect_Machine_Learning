"""
sp_series_temporales_avanzado.py

Caja de herramientas para Forecasting Avanzado.

1. FEATURE ENGINEERING
    crear_features_temporales()
    crear_lags()
    crear_medias_moviles()

2. TRAIN / TEST
    train_test_ts_multivariante()
    mostrar_periodos_train_test()
    comprobar_alineacion_temporal()

3. ENTRENAMIENTO
    entrenar_sarimax()
    resumen_modelo_ts()
    entrenar_modelos_base_ts()
    entrenar_modelo_final_sarimax()
    guardar_orden_modelo()

4. FORECAST
    forecast_sarimax()
    forecast_intervalos()
    visualizar_forecast()
    visualizar_forecast_intervalos()
    forecast_futuro_multivariante()
    crear_dataframe_forecast()
    exportar_forecast_csv()

5. EVALUACIÓN
    comparar_modelos_ts()
    mejor_modelo_ts()
    ranking_modelos_ts()
    resumen_metricas_ts()
    evaluar_forecast()
    visualizar_errores_forecast()
    evaluar_multiple_forecast()

6. OPTIMIZACIÓN
    buscar_mejor_sarimax()
    buscar_mejor_seasonal_order()
    grid_search_sarimax()
    comparar_aic_modelos()
    mostrar_mejor_modelo()

7. INTERPRETACIÓN
    importancia_variables_sarimax()
    interpretar_coeficientes_sarimax()
    analizar_residuos_sarimax()
    resumen_forecast()
    detectar_tendencia_forecast()
    comparar_forecast_real()
    informe_final_forecast()
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

from statsmodels.graphics.tsaplots import (plot_acf,plot_pacf)

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX


# Para que se muestren todas las columnas al inspeccionar los DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)




# ==========================================================
# 1. FEATURE ENGINEERING
# ==========================================================

def crear_features_temporales(
    df,
    fecha_col
):
    """
    Genera variables temporales a partir de una fecha.

    Crea automáticamente:

    - anio
    - trimestre
    - mes
    - semana
    - dia
    - dia_semana

    Muy útil para modelos que utilizan variables
    explicativas (SARIMAX, Prophet, XGBoost, etc.).

    Parameters
    ----------
    df : DataFrame
        Dataset original.

    fecha_col : str
        Nombre de la columna fecha.

    Returns
    -------
    DataFrame
        DataFrame con nuevas variables temporales.
    """

    df = df.copy()

    df[fecha_col] = pd.to_datetime(
        df[fecha_col]
    )

    df["anio"] = (
        df[fecha_col].dt.year
    )

    df["trimestre"] = (
        df[fecha_col].dt.quarter
    )

    df["mes"] = (
        df[fecha_col].dt.month
    )

    df["semana"] = (
        df[fecha_col]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["dia"] = (
        df[fecha_col].dt.day
    )

    df["dia_semana"] = (
        df[fecha_col].dt.dayofweek
    )

    return df


def crear_lags(
    df,
    target,
    lags=(1, 3, 6, 12)
):
    """
    Genera variables retardadas (lags).

    Los lags son una de las técnicas más utilizadas
    en forecasting porque permiten al modelo conocer
    el comportamiento pasado de la variable objetivo.

    Ejemplo:

    lag_1  -> valor del periodo anterior
    lag_3  -> valor de hace 3 periodos
    lag_12 -> valor de hace 12 periodos

    Parameters
    ----------
    df : DataFrame

    target : str
        Variable objetivo.

    lags : iterable
        Retardos a crear.

    Returns
    -------
    DataFrame
        DataFrame con columnas lag.
    """

    df = df.copy()

    for lag in lags:

        df[f"lag_{lag}"] = (
            df[target]
            .shift(lag)
        )

    return df


def crear_medias_moviles(
    df,
    target,
    ventanas=(3, 6, 12)
):
    """
    Genera medias móviles.

    Las medias móviles suavizan la serie temporal
    y ayudan a capturar tendencias de corto,
    medio y largo plazo.

    Ejemplos:

    rolling_3  -> media últimos 3 periodos
    rolling_6  -> media últimos 6 periodos
    rolling_12 -> media últimos 12 periodos

    Parameters
    ----------
    df : DataFrame

    target : str
        Variable objetivo.

    ventanas : iterable
        Ventanas para calcular medias móviles.

    Returns
    -------
    DataFrame
        DataFrame con nuevas variables rolling.
    """

    df = df.copy()

    for ventana in ventanas:

        df[f"rolling_{ventana}"] = (
            df[target]
            .rolling(window=ventana)
            .mean()
        )

    return df


# ==========================================================
# 2. TRAIN TEST
# ==========================================================

def train_test_ts_multivariante(
    X,
    y,
    test_size=0.2
):
    """
    División temporal para forecasting multivariante.

    A diferencia del Machine Learning clásico, las observaciones
    temporales nunca deben mezclarse, ya que el objetivo consiste
    en predecir valores futuros utilizando únicamente información
    del pasado.

    Esta función divide simultáneamente:

    - Variables explicativas (X)
    - Variable objetivo (y)

    manteniendo el orden cronológico original.

    Ejemplo
    --------
    X = df[
        [
            'horas_estudio_medias',
            'asistencia_media',
            'tasa_aprobados'
        ]
    ]

    y = df['nota_media']

    X_train, X_test, y_train, y_test = (
        train_test_ts_multivariante(
            X,
            y,
            test_size=0.2
        )
    )

    Parameters
    ----------
    X : DataFrame
        Variables explicativas.

    y : Series
        Variable objetivo.

    test_size : float
        Porcentaje reservado para test.
        Por defecto 0.2 (20%).

    Returns
    -------
    X_train : DataFrame
        Variables explicativas entrenamiento.

    X_test : DataFrame
        Variables explicativas test.

    y_train : Series
        Objetivo entrenamiento.

    y_test : Series
        Objetivo test.
    """

    corte = int(
        len(X) * (1 - test_size)
    )

    X_train = X.iloc[:corte]

    X_test = X.iloc[corte:]

    y_train = y.iloc[:corte]

    y_test = y.iloc[corte:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def mostrar_periodos_train_test(
    y_train,
    y_test
):
    """
    Muestra el rango temporal utilizado
    en entrenamiento y test.

    Función auxiliar útil para validar que la
    división temporal se ha realizado correctamente.

    Parameters
    ----------
    y_train : Series

    y_test : Series

    Returns
    -------
    None
    """

    print("=" * 60)

    print("TRAIN")

    print(
        f"Inicio: {y_train.index.min()}"
    )

    print(
        f"Fin:    {y_train.index.max()}"
    )

    print(
        f"Observaciones: {len(y_train)}"
    )

    print("=" * 60)

    print("TEST")

    print(
        f"Inicio: {y_test.index.min()}"
    )

    print(
        f"Fin:    {y_test.index.max()}"
    )

    print(
        f"Observaciones: {len(y_test)}"
    )

    print("=" * 60)


def comprobar_alineacion_temporal(
    X,
    y
):
    """
    Comprueba que X e y tienen exactamente
    las mismas fechas.

    Esta validación es especialmente importante
    después de crear:

    - lags
    - rolling means
    - transformaciones temporales

    ya que pueden provocar desalineaciones.

    Parameters
    ----------
    X : DataFrame

    y : Series

    Returns
    -------
    bool
        True si están alineados.
    """

    alineado = X.index.equals(
        y.index
    )

    if alineado:

        print(
            "✓ X e y están alineados temporalmente."
        )

    else:

        print(
            "✗ X e y NO están alineados temporalmente."
        )

    return alineado

# ==========================================================
# 3. ENTRENAMIENTO
# ==========================================================

def entrenar_sarimax(
    y_train,
    exog_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
):
    """
    Entrena un modelo SARIMAX.

    SARIMAX (Seasonal AutoRegressive Integrated Moving Average
    with eXogenous variables) extiende SARIMA permitiendo
    incorporar variables externas (exógenas) que ayudan a
    explicar el comportamiento de la serie objetivo.

    Ejemplo:

    y_train:
        nota_media

    exog_train:
        horas_estudio_medias
        asistencia_media
        tasa_aprobados

    Parameters
    ----------
    y_train : Series
        Variable objetivo.

    exog_train : DataFrame
        Variables exógenas.

    order : tuple
        Parámetros (p,d,q).

    seasonal_order : tuple
        Parámetros estacionales
        (P,D,Q,s).

    Returns
    -------
    modelo_entrenado
        Modelo ajustado.
    """

    modelo = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order
    )

    resultado = modelo.fit(
        disp=False
    )

    return resultado


def resumen_modelo_ts(
    modelo
):
    """
    Muestra el resumen estadístico
    del modelo.

    Parameters
    ----------
    modelo :
        Modelo entrenado.

    Returns
    -------
    summary
    """

    return modelo.summary()


def entrenar_modelos_base_ts(
    train,
    exog_train=None
):
    """
    Entrena modelos base de forecasting
    para comparación rápida.

    Si exog_train es None:
        - ARIMA(1,1,1)
        - SARIMA(1,1,1)(1,1,1,12)

    Si exog_train existe:
        - SARIMAX(1,1,1)(1,1,1,12)

    Parameters
    ----------
    train : Series

    exog_train : DataFrame, opcional

    Returns
    -------
    dict
        Diccionario con modelos.
    """

    resultados = {}

    if exog_train is None:

        resultados["ARIMA"] = (
            ARIMA(
                train,
                order=(1, 1, 1)
            ).fit()
        )

        resultados["SARIMA"] = (
            SARIMAX(
                train,
                order=(1, 1, 1),
                seasonal_order=(
                    1,
                    1,
                    1,
                    12
                )
            ).fit(
                disp=False
            )
        )

    else:

        resultados["SARIMAX"] = (
            SARIMAX(
                train,
                exog=exog_train,
                order=(1, 1, 1),
                seasonal_order=(
                    1,
                    1,
                    1,
                    12
                )
            ).fit(
                disp=False
            )
        )

    return resultados


def entrenar_modelo_final_sarimax(
    y,
    exog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
):
    """
    Entrena el modelo final utilizando
    toda la información disponible.

    Esta función se utiliza después de:

    - Evaluar modelos.
    - Optimizar hiperparámetros.
    - Seleccionar el mejor SARIMAX.

    Parameters
    ----------
    y : Series
        Variable objetivo completa.

    exog : DataFrame
        Variables exógenas completas.

    order : tuple
        Parámetros (p,d,q).

    seasonal_order : tuple
        Parámetros estacionales.

    Returns
    -------
    modelo_final
    """

    modelo = SARIMAX(
        endog=y,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order
    )

    resultado = modelo.fit(
        disp=False)
    return resultado


def guardar_orden_modelo(
    order,
    seasonal_order=None
):
    """
    Guarda de forma estructurada
    la configuración del modelo.

    Parameters
    ----------
    order : tuple

    seasonal_order : tuple, opcional

    Returns
    -------
    dict
    """

    resultado = {
        "order": order
    }

    if seasonal_order is not None:

        resultado[
            "seasonal_order"
        ] = seasonal_order

    return resultado


# ==========================================================
# 4. FORECAST
# ==========================================================

def forecast_sarimax(
    modelo,
    exog_future,
    pasos
):
    """
    Realiza predicciones utilizando un modelo SARIMAX.

    A diferencia de ARIMA y SARIMA, SARIMAX requiere
    proporcionar los valores futuros de las variables
    exógenas.

    Ejemplo:

    exog_future = pd.DataFrame(
        {
            'horas_estudio_medias': [...],
            'asistencia_media': [...],
            'tasa_aprobados': [...]
        }
    )

    Parameters
    ----------
    modelo :
        Modelo SARIMAX entrenado.

    exog_future : DataFrame
        Variables exógenas futuras.

    pasos : int
        Número de periodos a predecir.

    Returns
    -------
    Series
        Predicciones futuras.
    """

    return modelo.forecast(
        steps=pasos,
        exog=exog_future
    )


def forecast_intervalos(
    modelo,
    pasos,
    exog_future=None,
    alpha=0.05
):
    """
    Genera predicciones e intervalos de confianza.

    Además de la predicción puntual, devuelve
    los límites inferior y superior esperados.

    Ejemplo:

    forecast
    82.5

    lower
    81.7

    upper
    83.4

    Parameters
    ----------
    modelo :
        Modelo entrenado.

    pasos : int
        Número de periodos futuros.

    exog_future : DataFrame, opcional
        Variables exógenas futuras.

    alpha : float
        Nivel de significación.

    Returns
    -------
    DataFrame
        Forecast + intervalos.
    """

    pred = modelo.get_forecast(
        steps=pasos,
        exog=exog_future
    )

    resultado = pd.DataFrame(
        {
            "forecast":
                pred.predicted_mean,
            "lower":
                pred.conf_int().iloc[:, 0],
            "upper":
                pred.conf_int().iloc[:, 1]
        }
    )

    return resultado


def visualizar_forecast(
    y_train,
    y_test,
    y_pred,
    titulo="Forecast"
):
    """
    Representa visualmente:

    - Train
    - Test
    - Forecast

    Muy útil para comprobar si el modelo
    sigue correctamente la tendencia de
    la serie temporal.

    Parameters
    ----------
    y_train : Series

    y_test : Series

    y_pred : Series

    titulo : str

    Returns
    -------
    None
    """

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        y_train.index,
        y_train,
        label="Train"
    )

    plt.plot(
        y_test.index,
        y_test,
        label="Real"
    )

    plt.plot(
        y_pred.index,
        y_pred,
        label="Forecast"
    )

    plt.title(titulo)

    plt.legend()

    plt.grid()

    plt.show()


def visualizar_forecast_intervalos(
    y,
    forecast_df,
    titulo="Forecast con intervalos"
):
    """
    Representa:

    - Serie histórica
    - Forecast
    - Intervalos de confianza

    Es la visualización más habitual
    en forecasting profesional.

    Parameters
    ----------
    y : Series
        Serie histórica.

    forecast_df : DataFrame
        Resultado de forecast_intervalos().

    titulo : str

    Returns
    -------
    None
    """

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        y.index,
        y,
        label="Histórico"
    )

    plt.plot(
        forecast_df.index,
        forecast_df["forecast"],
        label="Forecast"
    )

    plt.fill_between(
        forecast_df.index,
        forecast_df["lower"],
        forecast_df["upper"],
        alpha=0.2
    )

    plt.title(titulo)

    plt.legend()

    plt.grid()

    plt.show()


def forecast_futuro_multivariante(
    modelo,
    exog_future
):
    """
    Genera forecast futuro completo para
    modelos SARIMAX.

    La longitud de la predicción se obtiene
    automáticamente del número de filas de
    exog_future.

    Parameters
    ----------
    modelo :
        Modelo SARIMAX entrenado.

    exog_future : DataFrame
        Variables futuras conocidas o estimadas.

    Returns
    -------
    Series
        Predicciones futuras.
    """

    return modelo.forecast(
        steps=len(exog_future),
        exog=exog_future
    )

def crear_dataframe_forecast(
    fechas_futuras,
    forecast
):
    """
    Convierte una predicción temporal en un DataFrame.

    Muy útil para:

    - Exportar resultados.
    - Crear dashboards.
    - Generar informes.
    - Comparar forecasts.

    Parameters
    ----------
    fechas_futuras : iterable
        Fechas futuras asociadas a la predicción.

    forecast : Series o array
        Predicciones generadas por el modelo.

    Returns
    -------
    DataFrame
        DataFrame con:

        - fecha
        - forecast
    """

    forecast_df = pd.DataFrame(
        {
            "fecha": fechas_futuras,
            "forecast": forecast
        }
    )

    return forecast_df

def exportar_forecast_csv(
    forecast_df,
    ruta
):
    """
    Exporta un forecast a formato CSV.

    Permite guardar una predicción para:

    - Informes.
    - Power BI.
    - Excel.
    - Dashboards.
    - Compartir resultados.

    Parameters
    ----------
    forecast_df : DataFrame
        DataFrame generado mediante
        crear_dataframe_forecast().

    ruta : str
        Ruta de salida.

    Returns
    -------
    None
    """

    forecast_df.to_csv(
        ruta,
        index=False
    )

    print("=" * 60)

    print(
        "FORECAST EXPORTADO"
    )

    print(
        f"Ruta: {ruta}"
    )

    print(
        f"Filas: {forecast_df.shape[0]}"
    )

    print("=" * 60)


# ==========================================================
# 5. EVALUACIÓN
# ==========================================================

def comparar_modelos_ts(
    resultados
):
    """
    Compara métricas de varios modelos de forecasting.

    Permite comparar fácilmente modelos como:

    - ARIMA
    - SARIMA
    - SARIMAX
    - Prophet

    Parameters
    ----------
    resultados : dict

        Ejemplo:

        {
            "ARIMA": {
                "MAPE": 1.2,
                "RMSE": 2.1
            },
            "SARIMA": {
                "MAPE": 0.9,
                "RMSE": 1.8
            }
        }

    Returns
    -------
    DataFrame
        Tabla comparativa ordenada
        por MAPE ascendente.
    """

    df = pd.DataFrame(
        resultados
    ).T

    if "MAPE" in df.columns:

        df = df.sort_values(
            by="MAPE"
        )

    return df


def mejor_modelo_ts(
    resultados,
    metrica="MAPE"
):
    """
    Identifica automáticamente el mejor modelo.

    Parameters
    ----------
    resultados : dict
        Diccionario de métricas.

    metrica : str
        Métrica utilizada para decidir.

    Returns
    -------
    tuple

        (
            nombre_modelo,
            resultado_metrica
        )
    """

    mejor_modelo = min(
        resultados,
        key=lambda x:
        resultados[x][metrica]
    )

    return (
        mejor_modelo,
        resultados[
            mejor_modelo
        ][metrica]
    )


def ranking_modelos_ts(
    resultados,
    metrica="MAPE"
):
    """
    Genera un ranking de modelos.

    Parameters
    ----------
    resultados : dict

    metrica : str

    Returns
    -------
    DataFrame
    """

    ranking = (
        pd.DataFrame(
            resultados
        )
        .T
        .sort_values(
            by=metrica
        )
    )

    return ranking


def resumen_metricas_ts(
    metricas
):
    """
    Muestra un resumen amigable
    de las métricas de forecast.

    Parameters
    ----------
    metricas : dict

    Returns
    -------
    None
    """

    print("=" * 60)

    print("MÉTRICAS FORECAST")

    print("=" * 60)

    for k, v in metricas.items():

        print(
            f"{k:<10}: {v:.4f}"
        )

    print("=" * 60)


def evaluar_forecast(
    y_real,
    y_pred
):
    """
    Wrapper rápido para obtener las
    métricas más habituales de forecasting.

    Métricas:

    - MAE
    - MSE
    - RMSE
    - MAPE
    - R²

    Parameters
    ----------
    y_real : Series

    y_pred : Series

    Returns
    -------
    dict
    """

    y_real = np.array(y_real)

    y_pred = np.array(y_pred)

    mse = mean_squared_error(
        y_real,
        y_pred
    )

    mask = y_real != 0

    mape = np.mean(
        np.abs(
            (
                y_real[mask]
                - y_pred[mask]
            )
            /
            y_real[mask]
        )
    ) * 100

    return {
        "MAE": mean_absolute_error(
            y_real,
            y_pred
        ),
        "MSE": mse,
        "RMSE": np.sqrt(
            mse
        ),
        "MAPE": mape,
        "R2": r2_score(
            y_real,
            y_pred
        )
    }


def visualizar_errores_forecast(
    y_real,
    y_pred,
    titulo="Errores Forecast"
):
    """
    Representa los errores del forecast.

    Permite visualizar:

    error = real - predicción

    Parameters
    ----------
    y_real : Series

    y_pred : Series

    titulo : str

    Returns
    -------
    Series
        Errores.
    """

    errores = (
        np.array(y_real)
        -
        np.array(y_pred)
    )

    plt.figure(
        figsize=(12, 4)
    )

    plt.plot(
        errores
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.title(
        titulo
    )

    plt.grid()

    plt.show()

    return errores


def evaluar_multiple_forecast(
    y_real,
    forecasts
):
    """
    Evalúa múltiples modelos de forecasting
    utilizando las mismas observaciones reales.

    Permite comparar rápidamente modelos como:

    - ARIMA
    - SARIMA
    - SARIMAX
    - Prophet

    Ejemplo
    --------

    forecasts = {
        "ARIMA": pred_arima,
        "SARIMA": pred_sarima,
        "SARIMAX": pred_sarimax
    }

    resultados = evaluar_multiple_forecast(
        y_test,
        forecasts
    )

    Parameters
    ----------
    y_real : Series
        Valores observados.

    forecasts : dict

        Diccionario con formato:

        {
            "ARIMA": pred1,
            "SARIMA": pred2,
            "SARIMAX": pred3
        }

    Returns
    -------
    DataFrame

        Tabla comparativa con:

        - MAE
        - MSE
        - RMSE
        - MAPE
        - R2

        Ordenada por MAPE ascendente.
    """

    resultados = {}

    for nombre_modelo, predicciones in forecasts.items():

        metricas = evaluar_forecast(
            y_real,
            predicciones
        )

        resultados[
            nombre_modelo
        ] = metricas

    resultados = (
        pd.DataFrame(
            resultados
        )
        .T
        .sort_values(
            by="MAPE"
        )
    )

    return resultados

# ==========================================================
# 6. OPTIMIZACIÓN
# ==========================================================

def buscar_mejor_sarimax(
    y_train,
    exog_train,
    p_range=(0, 3),
    d_range=(0, 2),
    q_range=(0, 3),
    seasonal_order=(1, 1, 1, 12)
):
    """
    Busca la mejor configuración SARIMAX utilizando AIC.

    Se prueban múltiples combinaciones de:

    - p
    - d
    - q

    manteniendo fija la parte estacional.

    Parameters
    ----------
    y_train : Series
        Variable objetivo.

    exog_train : DataFrame
        Variables exógenas.

    p_range : tuple
        Rango de p.

    d_range : tuple
        Rango de d.

    q_range : tuple
        Rango de q.

    seasonal_order : tuple
        Parámetros SARIMA:
        (P,D,Q,s)

    Returns
    -------
    tuple

        (
            mejor_order,
            mejor_aic
        )
    """

    mejor_aic = np.inf

    mejor_order = None

    for p, d, q in itertools.product(
        range(*p_range),
        range(*d_range),
        range(*q_range)
    ):

        try:

            modelo = SARIMAX(
                y_train,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=seasonal_order
            )

            resultado = modelo.fit(
                disp=False
            )

            if resultado.aic < mejor_aic:

                mejor_aic = resultado.aic

                mejor_order = (
                    p,
                    d,
                    q
                )

        except Exception:

            pass

    return (
        mejor_order,
        mejor_aic
    )


def buscar_mejor_seasonal_order(
    y_train,
    exog_train,
    order=(1, 1, 1),
    seasonal_period=12
):
    """
    Busca la mejor combinación estacional
    para SARIMAX.

    Explora:

    (P,D,Q,s)

    utilizando AIC.

    Parameters
    ----------
    y_train : Series

    exog_train : DataFrame

    order : tuple
        Parte no estacional.

    seasonal_period : int
        Periodicidad.

        12 = mensual
        4  = trimestral
        7  = semanal

    Returns
    -------
    tuple

        (
            seasonal_order,
            mejor_aic
        )
    """

    mejor_aic = np.inf

    mejor_seasonal = None

    for P, D, Q in itertools.product(
        range(3),
        range(2),
        range(3)
    ):

        try:

            seasonal_order = (
                P,
                D,
                Q,
                seasonal_period
            )

            modelo = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order
            )

            resultado = modelo.fit(
                disp=False
            )

            if resultado.aic < mejor_aic:

                mejor_aic = resultado.aic

                mejor_seasonal = (
                    seasonal_order
                )

        except Exception:

            pass

    return (
        mejor_seasonal,
        mejor_aic
    )


def grid_search_sarimax(
    y_train,
    exog_train,
    p_range=(0,3),
    d_range=(0,2),
    q_range=(0,3),
    P_range=(0,2),
    D_range=(0,2),
    Q_range=(0,2),
    seasonal_period=12
):
    """
    Grid Search completo para SARIMAX.

    Busca simultáneamente:

    - p,d,q
    - P,D,Q,s

    utilizando AIC como criterio de selección.

    Parameters
    ----------
    y_train : Series

    exog_train : DataFrame

    seasonal_period : int

    Returns
    -------
    dict
    """

    mejor_aic = np.inf

    mejor_order = None

    mejor_seasonal = None

    for p, d, q in itertools.product(
        range(*p_range),
        range(*d_range),
        range(*q_range)
    ):

        for P, D, Q in itertools.product(
            range(*P_range),
            range(*D_range),
            range(*Q_range)
        ):

            try:

                seasonal_order = (
                    P,
                    D,
                    Q,
                    seasonal_period
                )

                modelo = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=(p, d, q),
                    seasonal_order=seasonal_order
                )

                resultado = modelo.fit(
                    disp=False
                )

                if resultado.aic < mejor_aic:

                    mejor_aic = resultado.aic

                    mejor_order = (
                        p,
                        d,
                        q
                    )

                    mejor_seasonal = (
                        seasonal_order
                    )

            except Exception:

                pass

    return {
        "order": mejor_order,
        "seasonal_order": mejor_seasonal,
        "AIC": mejor_aic
    }


def comparar_aic_modelos(
    modelos
):
    """
    Compara varios modelos utilizando AIC
    (Akaike Information Criterion).

    Un AIC más bajo indica un mejor equilibrio
    entre ajuste y complejidad del modelo.

    Ejemplo
    --------

    modelos = {
        "ARIMA": modelo_arima,
        "SARIMA": modelo_sarima,
        "SARIMAX": modelo_sarimax
    }

    comparar_aic_modelos(
        modelos
    )

    Parameters
    ----------
    modelos : dict

        Diccionario con modelos entrenados.

    Returns
    -------
    DataFrame

        Tabla ordenada por AIC ascendente.
    """

    resultados = []

    for nombre, modelo in modelos.items():

        resultados.append(
            {
                "Modelo": nombre,
                "AIC": modelo.aic
            }
        )

    resultados = pd.DataFrame(
        resultados
    )

    resultados = resultados.sort_values(
        by="AIC"
    )

    resultados = resultados.reset_index(
        drop=True
    )

    return resultados

def mostrar_mejor_modelo(
    resultado_busqueda
):
    """
    Muestra el mejor modelo encontrado
    durante un proceso de optimización.

    Compatible con:

    - buscar_mejor_arima()
    - buscar_mejor_sarimax()
    - grid_search_sarimax()

    Ejemplo
    --------

    resultado = {

        "order": (2,1,2),

        "seasonal_order": (1,1,1,12),

        "AIC": 25.42

    }

    mostrar_mejor_modelo(
        resultado
    )

    Parameters
    ----------
    resultado_busqueda : dict

        Diccionario con la configuración
        óptima obtenida durante la búsqueda.

    Returns
    -------
    None
    """

    print("=" * 70)

    print(
        "MEJOR MODELO ENCONTRADO"
    )

    print("=" * 70)

    for clave, valor in (
        resultado_busqueda.items()
    ):

        print(
            f"{clave:<20}: {valor}"
        )

    print("=" * 70)


# ==========================================================
# 7. INTERPRETACIÓN
# ==========================================================

def importancia_variables_sarimax(
    modelo
):
    """
    Extrae los coeficientes de las variables exógenas
    utilizadas por un modelo SARIMAX.

    Permite interpretar qué variables tienen un mayor
    impacto sobre la variable objetivo.

    Ejemplo:

    horas_estudio_medias    0.42
    asistencia_media        0.31
    tasa_aprobados          0.18

    Parameters
    ----------
    modelo :
        Modelo SARIMAX entrenado.

    Returns
    -------
    DataFrame
        Variables y coeficientes.
    """

    parametros = modelo.params

    resultados = pd.DataFrame(
        {
            "Variable": parametros.index,
            "Coeficiente": parametros.values
        }
    )

    return resultados


def interpretar_coeficientes_sarimax(
    modelo
):
    """
    Ordena los coeficientes por
    importancia absoluta.
    """

    coeficientes = modelo.params

    resultado = pd.DataFrame(
        {
            "Coeficiente": coeficientes
        }
    )

    resultado["Importancia"] = np.abs(
        resultado["Coeficiente"]
    )

    resultado = resultado.sort_values(
        by="Importancia",
        ascending=False
    )

    return resultado


def analizar_residuos_sarimax(
    modelo
):
    """
    Analiza los residuos de un modelo SARIMAX.

    Un buen modelo debería producir residuos
    similares a ruido blanco:

    - Media cercana a 0.
    - Sin tendencia.
    - Sin autocorrelación.
    - Distribución aproximadamente normal.

    Se representan:

    - Serie temporal de residuos.
    - Histograma.
    - Función de autocorrelación (ACF).

    Parameters
    ----------
    modelo :
        Modelo SARIMAX entrenado.

    Returns
    -------
    Series
        Residuos del modelo.
    """

    residuos = modelo.resid

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    residuos.plot(
        ax=ax[0],
        title="Residuos"
    )

    residuos.hist(
        bins=20,
        ax=ax[1]
    )

    ax[1].set_title(
        "Histograma residuos"
    )

    plt.tight_layout()

    plt.show()

    plot_acf(
        residuos.dropna()
    )

    plt.show()

    return residuos

def resumen_forecast(
    forecast_df
):
    """
    Genera estadísticas descriptivas
    del forecast.

    Útil para informes rápidos.

    Parameters
    ----------
    forecast_df : DataFrame

        DataFrame generado mediante
        crear_dataframe_forecast().

    Returns
    -------
    dict
    """

    columna_forecast = (
        forecast_df.columns[-1]
    )

    return {

        "Observaciones":
            len(forecast_df),

        "Forecast_Min":
            forecast_df[
                columna_forecast
            ].min(),

        "Forecast_Medio":
            forecast_df[
                columna_forecast
            ].mean(),

        "Forecast_Max":
            forecast_df[
                columna_forecast
            ].max()

    }

def detectar_tendencia_forecast(
    forecast
):
    """
    Detecta la tendencia general
    del forecast.

    Parameters
    ----------
    forecast : Series

    Returns
    -------
    str

        - Creciente
        - Decreciente
        - Estable
    """

    inicio = forecast.iloc[0]

    fin = forecast.iloc[-1]

    diferencia = fin - inicio

    if diferencia > 0:

        return "Creciente"

    elif diferencia < 0:

        return "Decreciente"

    else:

        return "Estable"

def comparar_forecast_real(
    y_real,
    y_pred
):
    """
    Construye una tabla comparativa
    entre valores reales y forecast.

    Permite analizar errores
    observación a observación.

    Parameters
    ----------
    y_real : Series

    y_pred : Series

    Returns
    -------
    DataFrame
    """

    resultado = pd.DataFrame(
        {
            "Real": y_real,
            "Forecast": y_pred
        }
    )

    resultado["Error"] = (
        resultado["Real"]
        -
        resultado["Forecast"]
    )

    resultado[
        "Error_Absoluto"
    ] = np.abs(
        resultado["Error"]
    )

    return resultado

def informe_final_forecast(
    metricas,
    forecast,
    modelo="SARIMAX"
):
    """
    Genera un informe resumido del modelo
    y del forecast realizado.

    Ideal para dashboards,
    presentaciones o documentación.

    Parameters
    ----------
    metricas : dict

        Salida de:

        evaluar_forecast()

    forecast : Series

        Forecast generado.

    modelo : str

        Nombre del modelo.

    Returns
    -------
    dict
    """

    informe = {

        "Modelo":
            modelo,

        "MAPE":
            metricas.get(
                "MAPE"
            ),

        "RMSE":
            metricas.get(
                "RMSE"
            ),

        "MAE":
            metricas.get(
                "MAE"
            ),

        "Tendencia":
            detectar_tendencia_forecast(
                forecast
            ),

        "Forecast_Inicial":
            forecast.iloc[0],

        "Forecast_Final":
            forecast.iloc[-1]

    }

    return informe