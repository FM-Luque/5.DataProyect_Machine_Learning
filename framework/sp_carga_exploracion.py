
"""
carga_exploracion.py
================
Caja de herramientas de Fase 1 — Carga, validación y exploración inicial.

Dos formas de usarlo:

  1) Como punto de entrada único, si el proyecto solo tiene UNA fuente
     de datos (sin necesidad de merge): leer_csv()/leer_excel() cargan
     directamente, y exploracion_inicial() da la primera vista.

  2) Justo DESPUÉS de integracion.py, si el proyecto necesita unir dos
     o más fuentes: integracion.py se encarga de cargar y hacer el
     merge; este módulo valida el resultado (comprobar_claves() antes
     del merge, verificar_union() después) y da la vista inicial del
     DataFrame ya integrado.

Este módulo NO limpia ni transforma datos (eso vive en
sp_limpieza_trasfromacion.py, Fase 2). Solo carga, valida y explora.

  leer_csv()             -> lee un csv, avisa si el archivo no existe
  leer_excel()           -> lee una hoja de un excel, avisa si no existe
  leer_otros_formatos()  -> lee parquet,json,pickle
  exploracion_inicial()  -> vista rápida: sample, info, nulos,
                             duplicados, distribución de categóricas
"""

import pandas as pd
from IPython.display import display

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)


# ============================================================================
# 0. LECTURA DE ARCHIVOS
# ============================================================================

def leer_csv(ruta, parse_dates=None, **kwargs):
    """
    Lee un archivo csv y avisa si no lo encuentra, en vez de dar
    un error de Python difícil de entender.

    Parameters
    ----------
    ruta : str
        Ruta al archivo csv.
    parse_dates : list, opcional
        Columnas a interpretar como fecha (ej. ['fecha_pedido']).
    kwargs :
        Argumentos adicionales que acepta pd.read_csv().

    Returns
    -------
    DataFrame, o None si el archivo no existe.
    """
    try:
        df = pd.read_csv(ruta, parse_dates=parse_dates, **kwargs)
    except FileNotFoundError:
        print(f'No se ha encontrado el archivo: {ruta}')
        print('Revisa que la ruta sea correcta.')
        return None

    print('=' * 70)
    print('CSV CARGADO')
    print('=' * 70)
    print(f'Archivo: {ruta}')
    print(f'Filas:   {df.shape[0]}')
    print(f'Columnas: {df.shape[1]}')
    print('=' * 70)

    return df


def leer_excel(ruta, hoja=0, parse_dates=None, **kwargs):
    """
    Lee una hoja de un archivo Excel y avisa si no lo encuentra.

    Parameters
    ----------
    ruta : str
        Ruta al archivo .xlsx.
    hoja : str o int
        Nombre o índice de la hoja a leer (por defecto la primera, 0).
    parse_dates : list, opcional
        Columnas a interpretar como fecha.
    kwargs :
        Argumentos adicionales que acepta pd.read_excel().

    Returns
    -------
    DataFrame, o None si el archivo no existe.
    """
    try:
        df = pd.read_excel(ruta, sheet_name=hoja, parse_dates=parse_dates, **kwargs)
    except FileNotFoundError:
        print(f'No se ha encontrado el archivo: {ruta}')
        print('Revisa que la ruta sea correcta.')
        return None

    print('=' * 70)
    print('EXCEL CARGADO')
    print('=' * 70)
    print(f'Archivo: {ruta}')
    print(f'Hoja:    {hoja}')
    print(f'Filas:   {df.shape[0]}')
    print(f'Columnas: {df.shape[1]}')
    print('=' * 70)

    return df




# ============================================================================
# 1. EXPLORACIÓN INICIAL
# ============================================================================

def exploracion_inicial(df):
    """
    Realiza un análisis exploratorio inicial sobre un DataFrame dado.

    Este análisis incluye:
    - Muestra de 5 filas aleatorias
    - Información general del DataFrame (tipos de dato, nulos, etc.)
    - Porcentaje de valores nulos
    - Conteo de filas duplicadas
    - Distribución de valores para columnas categóricas

    Parameters
    ----------
    df : DataFrame
        DataFrame a analizar.

    Returns
    -------
    None
    """

    print('5 FILAS ALEATORIAS')
    display(df.sample(3).T)
    print('=' * 100)

    print('DIMENSIONES')
    print(f"El número de filas es: {df.shape[0]} filas ")
    print(f"El número de columnas es: {df.shape[1]} columnas")
    print('=' * 100)

    print('INFORMACION BASICA')
    print(df.info())
    print('=' * 100)

    print('COLUMNAS CON VALORES NULOS')
    print(df.isna().sum()[df.isna().sum() > 0])
    print('=' * 100)

    print('PORCENTAJES DE COLUMNAS CON VALORES NULOS')
    print(df.isna().mean()[df.isna().mean() > 0] * 100)
    print('=' * 100)

    print('VALORES DUPLICADOS ')
    print(df.duplicated().sum())
    print('=' * 100)

    print('CONTEO COLUMNAS CATEGORIAS ')

    for col in df.select_dtypes(include=['string','object', 'category']).columns:
        print(col.upper())
        print(df[col].value_counts())
    print('=' * 100)

    print('ESTADISTICOS CATEGÓRICOS')
    estadisticos_c = df.describe(include=['string','object', 'category']).T
    display(estadisticos_c)

    print('ESTADISTICOS NUMÉRICAS')
    estadisticos_n = df.describe(include='number').T
    display(estadisticos_n)



# ============================================================================
# 2. OTROS TIPOS DE ARCHIVO DE CARGA
# ============================================================================
def leer_html(
    url: str,
    *,
    tabla: int = 0,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Lee una tabla HTML desde una URL.

    Parameters
    ----------
    url : str
        Dirección web.
    tabla : int, default=0
        Índice de la tabla a cargar.
    verbose : bool, default=True
        Muestra información de carga.
    """

    tablas = pd.read_html(
        url,
        **kwargs
    )

    if not tablas:
        raise ValueError(f"No se encontraron tablas en: {url}")

    df = tablas[tabla]

    if verbose:
        print("=" * 70)
        print("HTML CARGADO")
        print("=" * 70)
        print(f"URL     : {url}")
        print(f"Tabla   : {tabla}")
        print(f"Filas   : {df.shape,}")
        print(f"Cols    : {df.shape[1]}")
        print("=" * 70)

    return df


def leer_parquet(
    ruta: str,
    *,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:

    df = pd.read_parquet(
        ruta,
        **kwargs
    )

    if verbose:
        print("=" * 70)
        print("PARQUET CARGADO")
        print("=" * 70)
        print(f"Archivo : {ruta}")
        print(f"Filas   : {df.shape,}")
        print(f"Cols    : {df.shape[1]}")
        print("=" * 70)

    return df


############################################################

def leer_json(
    ruta: str,
    *,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:

    df = pd.read_json(
        ruta,
        **kwargs
    )

    if verbose:
        print("=" * 70)
        print("JSON CARGADO")
        print("=" * 70)
        print(f"Archivo : {ruta}")
        print(f"Filas   : {df.shape,}")
        print(f"Cols    : {df.shape[1]}")
        print("=" * 70)

    return df


############################################################


def leer_pickle(
    ruta: str,
    *,
    verbose: bool = True
) -> pd.DataFrame:

    df = pd.read_pickle(ruta)

    if verbose:
        print("=" * 70)
        print("PICKLE CARGADO")
        print("=" * 70)
        print(f"Archivo : {ruta}")
        print(f"Filas   : {df.shape,}")
        print(f"Cols    : {df.shape[1]}")
        print("=" * 70)

    return df
