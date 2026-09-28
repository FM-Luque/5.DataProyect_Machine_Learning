"""
sp_imputacion_avanzada.py
================
Herramientas SITUACIONALES de imputación — no forman parte del flujo
estándar de 8 pasos de sp_limpieza_trasfromacion.py.

Por qué van aparte
------------------
La media, mediana, moda y valor constante (en sp_limpieza_trasfromacion.py)
son opciones razonables por defecto para casi cualquier columna. KNN e
interpolate() no lo son:

- imputar_knn(): necesita scikit-learn, solo funciona bien con columnas
  numéricas, y conviene escalarlas antes (si tienen escalas muy
  distintas, la distancia que usa KNN queda sesgada hacia la columna
  de valores más grandes).
- imputar_interpolacion(): solo tiene sentido en datos ORDENADOS
  (series temporales, o cualquier cosa con una secuencia lógica) —
  no es aplicable de forma genérica a una tabla cualquiera.

Por eso viven en `src/` (herramientas propias de este proyecto, usadas
solo cuando el caso concreto lo pide), no en `framework/` (donde están
las funciones de uso general en cualquier proyecto).

  imputar_knn()            -> imputación por vecinos más cercanos (KNN)
  imputar_interpolacion()  -> imputación por interpolación (datos ordenados)
"""



def imputar_knn(df, columnas, n_vecinos=5, escalar=True):
    """
    Imputa valores nulos usando KNNImputer (busca las filas más
    parecidas -vecinos- según el resto de columnas numéricas, y
    rellena el nulo con el valor medio de esos vecinos).

    Solo funciona con columnas NUMÉRICAS.

    Parameters
    ----------
    df : DataFrame
    columnas : list[str]
        Columnas numéricas a imputar.
    n_vecinos : int
        Número de vecinos a considerar (por defecto 5).
    escalar : bool
        Si True (por defecto), escala las columnas antes de calcular
        distancias (StandardScaler), para que ninguna columna domine
        el cálculo solo por tener valores más grandes. Se revierte el
        escalado al final, así que el resultado queda en la escala
        original.

    Returns
    -------
    DataFrame con los nulos de esas columnas imputados.

    Ejemplo
    -------
    imputar_knn(df, ['edad', 'ingresos_anuales'], n_vecinos=5)
    """
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler

    nuevo = df.copy()
    datos = nuevo[columnas]

    if escalar:
        scaler = StandardScaler()
        datos_escalados = scaler.fit_transform(datos)

        imputer = KNNImputer(n_neighbors=n_vecinos)
        datos_imputados = imputer.fit_transform(datos_escalados)

        datos_imputados = scaler.inverse_transform(datos_imputados)

    else:
        imputer = KNNImputer(n_neighbors=n_vecinos)
        datos_imputados = imputer.fit_transform(datos)

    nuevo[columnas] = datos_imputados

    return nuevo


def imputar_interpolacion(df, columnas, metodo='linear'):
    """
    Imputa valores nulos por interpolación: estima el valor que
    "encajaría" entre el dato anterior y el siguiente, siguiendo el
    orden actual de las filas.

    Solo tiene sentido si el DataFrame está ORDENADO de forma
    significativa (ej. por fecha) — si el orden de las filas es
    arbitrario, el resultado no tiene ningún sentido.

    Parameters
    ----------
    df : DataFrame
        Debe estar ya ordenado por la columna relevante (ej.
        df.sort_values('fecha') antes de llamar a esta función).
    columnas : list[str]
        Columnas numéricas a imputar.
    metodo : str
        Método de interpolación de pandas (por defecto 'linear').
        Otras opciones: 'time' (si el índice es de tipo fecha),
        'polynomial', 'spline'...

    Returns
    -------
    DataFrame con los nulos de esas columnas imputados.

    Ejemplo
    -------
    df_ordenado = df.sort_values('fecha_pedido')
    imputar_interpolacion(df_ordenado, ['ventas_dia'], metodo='linear')
    """
    nuevo = df.copy()

    for col in columnas:
        nuevo[col] = nuevo[col].interpolate(method=metodo)

    return nuevo
