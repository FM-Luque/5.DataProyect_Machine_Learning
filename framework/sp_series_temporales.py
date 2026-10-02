"""
sp_series_temporales.py

Caja de herramientas para Series Temporales.

Sigue el flujo del temario:

1. PREPARACIÓN
    check_continuity()
    fill_date_gaps()
    adf_test()
    descomponer_serie()
    graficos_autocorrelacion()

2. TRAIN / TEST
    train_test_ts()

3. ENTRENAMIENTO
    entrenar_arima()
    entrenar_sarima()

4. FORECAST
    forecast()

5. EVALUACIÓN
    metricas_ts()

6. OPTIMIZACIÓN
    buscar_mejor_arima()

7. ANÁLISIS
    analizar_residuos()

Los modelos siguen construyéndose con statsmodels.
Este módulo encapsula únicamente tareas repetitivas.
"""

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

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
# 1. Cargar dataframes
# ==========================================================

def cargar_csv(
    ruta,
    parse_dates=None,
    squeeze=False,
    **kwargs
):
    """    
    Lee un archivo csv y avisa si no lo encuentra, en vez de dar
    un error de Python difícil de entender.

    Parameters
    ----------
    ruta : str
        Ruta al archivo csv.
    parse_dates : list, opcional
        Columnas a interpretar como fecha (ej. ['fecha_pedido']).
    squeeze = True: 
        Elimina columna en y    
    kwargs :
        Argumentos adicionales que acepta pd.read_csv().

    Returns
    -------
    DataFrame, o None si el archivo no existe.
    """

    try:
        df = pd.read_csv(
            ruta,
            parse_dates=parse_dates,
            **kwargs
        )

    except FileNotFoundError:
        print(f'No se ha encontrado el archivo: {ruta}')
        print('Revisa que la ruta sea correcta.')
        return None

    if squeeze and df.shape[1] == 1:
        df = df.squeeze()

    print('=' * 70)
    print('CSV CARGADO')
    print('=' * 70)
    print(f'Archivo: {ruta}')
    print(f'Filas:   {df.shape[0]}')

    if isinstance(df, pd.DataFrame):
        print(f'Columnas: {df.shape[1]}')
    else:
        print('Columnas: 1 (Series)')

    print('=' * 70)

    return df


# ==========================================================
# 1. PREPARACIÓN
# ==========================================================

def check_continuity(
    df,
    fecha_col
):
    """
    Comprueba continuidad temporal, si hay huecos.
    """

    fechas = pd.to_datetime(df[fecha_col])

    diferencias = fechas.diff().dropna()

    return diferencias.value_counts()


def fill_date_gaps(
    df,
    fecha_col,
    frecuencia="D"
):
    """
    Rellena huecos temporales.
        # D -> Diario
        # W -> Semanal
        # MS -> Inicio de mes (Month Start)
        # M -> Fin de mes (Month End)
        # QS -> Inicio de trimestre (Quarter Start)
        # Q -> Fin de trimestre (Quarter End)
         
        # YS -> Inicio de año (Year Start)
        # Y -> Fin de año (Year End)
         
        # H -> Horario
        # min -> Minutos
        # S -> Segundos
    """

    df = df.copy()

    df[fecha_col] = pd.to_datetime(
        df[fecha_col]
    )

    df = df.set_index(
        fecha_col
    )

    df = df.asfreq(frecuencia)

    return df


def adf_test(serie):
    """
    Test de Dickey-Fuller Aumentado (ADF): comprueba si la serie es
    estacionaria (media y varianza constantes en el tiempo).

    Interpretación:
    - p-value < 0.05  -> se rechaza H0, la serie ES estacionaria.
    - p-value >= 0.05 -> no se rechaza H0, la serie NO es
      estacionaria (probablemente necesite diferenciación, es decir,
      un valor de 'd' > 0 en ARIMA).
    """

    resultado = adfuller(
        serie.dropna()
    )

    estadistico = resultado[0]
    pvalue = resultado[1]

    print(f"ADF Statistic: {estadistico:.4f}")
    print(f"p-value:       {pvalue:.4f}")

    if pvalue < 0.05:
        print("-> La serie ES estacionaria (p < 0.05).")
    else:
        print("-> La serie NO es estacionaria (p >= 0.05).")
        print("   Valora diferenciarla (aumentar 'd' en ARIMA) antes de modelar.")

    return {
        "ADF Statistic": estadistico,
        "p-value": pvalue,
        "estacionaria": pvalue < 0.05
    }


def descomponer_serie(serie, modelo="additive", periodo=None):
    """
    Descompone la serie en tendencia, estacionalidad y residuo, y
    representa las 4 partes (serie original + las 3 componentes).

    Parameters
    ----------
    serie : Series con índice de fecha.
    modelo : str
        'additive' (por defecto) si la amplitud de la estacionalidad
        se mantiene constante en el tiempo, o 'multiplicative' si la
        amplitud crece/decrece junto con el nivel de la serie.
    periodo : int, opcional
        Nº de observaciones que forman un ciclo estacional (ej. 7
        para datos diarios con patrón semanal, 12 para datos
        mensuales con patrón anual). Si no se indica, se intenta
        inferir de la frecuencia del índice.

    Returns
    -------
    El objeto de descomposición de statsmodels (con .trend,
    .seasonal, .resid, .observed).
    """

    descomposicion = seasonal_decompose(
        serie.dropna(),
        model=modelo,
        period=periodo
    )

    fig = descomposicion.plot()
    fig.set_size_inches(10, 8)
    plt.tight_layout()
    plt.show()

    return descomposicion

def graficos_autocorrelacion(serie,lags=40):
    """
    Representa ACF (autocorrelación) y PACF (autocorrelación
    parcial) uno al lado del otro — se usan juntos para elegir a
    mano los órdenes p y q de un modelo ARIMA:

    - PACF ayuda a elegir 'p' (parte autorregresiva, AR).
    - ACF ayuda a elegir 'q' (parte de media móvil, MA).

    Parameters
    ----------
    serie : Series (idealmente ya estacionaria — aplica antes
        adf_test() y diferencia si hace falta).
    lags : int
        Número de retardos a mostrar (por defecto 40).
    """

    max_lags = min(
        lags,
        len(serie) // 2 - 1
    )

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    plot_acf(
        serie.dropna(),
        lags=max_lags,
        ax=ax[0]
    )

    plot_pacf(
        serie.dropna(),
        lags=max_lags,
        ax=ax[1]
    )

    plt.tight_layout()
    plt.show()


# ==========================================================
# 2. TRAIN TEST
# ==========================================================

def train_test_ts(
    serie,
    test_size=0.2
):
    """
    División temporal.
    """

    corte = int(
        len(serie) * (1 - test_size)
    )

    train = serie[:corte]
    test = serie[corte:]

    return train, test

# ==========================================================
# 3. ENTRENAMIENTO
# ==========================================================

def entrenar_arima(
    train,
    order=(1, 1, 1)
):
    """
    Entrena ARIMA.
    """

    modelo = ARIMA(
        train,
        order=order
    )

    resultado = modelo.fit()

    return resultado


def entrenar_sarima(
    train,
    order=(1,1,1),
    seasonal_order=(1,1,1,12)
):
    """
    Entrena SARIMA.
    """

    modelo = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order
    )

    resultado = modelo.fit()

    return resultado

# ==========================================================
# 4. FORECAST
# ==========================================================

def forecast(
    modelo,
    pasos
):
    """
    Predicción futura.
    pasos = numero meses predicion 
    """

    return modelo.forecast(
        steps=pasos
    )

# ==========================================================
# 5. EVALUACIÓN
# ==========================================================

def metricas_ts(
    y_real,
    y_pred
):
    """
    MAE, MSE, RMSE, MAPE, R²

    MAPE (Error Porcentual Absoluto Medio) es la métrica más
    habitual para comunicar el error de un forecast, porque se
    expresa en % y es fácil de interpretar sin conocer la escala de
    la serie ("nos equivocamos un 8% de media").

    Aviso: MAPE no está definida si algún valor real es 0 (división
    por cero) — en ese caso, esos puntos se excluyen del cálculo.
    """

    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(
        y_real,
        y_pred
    )

    # MAPE: excluir puntos donde y_real == 0 para evitar división por cero
    mask = y_real != 0
    if mask.sum() < len(y_real):
        print(f'Aviso: {len(y_real) - mask.sum()} valores reales son 0, excluidos del cálculo de MAPE.')

    mape = np.mean(
        np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])
    ) * 100

    return {
        "MAE": mean_absolute_error(
            y_real,
            y_pred
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAPE": mape,
        "R2": r2_score(
            y_real,
            y_pred
        )
    }

# ==========================================================
# 6. OPTIMIZACIÓN
# ==========================================================

def buscar_mejor_arima(
    serie,
    p_range=(0, 3),
    d_range=(0, 2),
    q_range=(0, 3)
):
    """
    Búsqueda por AIC.
    """

    mejor_aic = np.inf
    mejor_order = None

    for p, d, q in itertools.product(
        range(*p_range),
        range(*d_range),
        range(*q_range)
    ):

        try:

            modelo = ARIMA(
                serie,
                order=(p,d,q)
            )

            resultado = modelo.fit()

            if resultado.aic < mejor_aic:

                mejor_aic = resultado.aic
                mejor_order = (p,d,q)

        except Exception:
            pass

    return mejor_order, mejor_aic

# ==========================================================
# 7. ANÁLISIS
# ==========================================================

def analizar_residuos(
    modelo
):
    """
    Visualización de residuos.
    """

    residuos = modelo.resid

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12,5)
    )

    residuos.plot(
        ax=ax[0],
        title="Residuos"
    )

    residuos.hist(
        bins=30,
        ax=ax[1]
    )

    ax[1].set_title(
        "Histograma residuos"
    )

    plt.show()

    plot_acf(residuos)

    plt.show()

    return residuos