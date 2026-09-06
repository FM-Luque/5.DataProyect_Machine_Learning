"""
sp_limpieza_trasfromacion.py
================
Caja de herramientas de Fase 2 — Limpieza y Transformación.

Sigue el flujo de 10 pasos:

  0. Exploración inicial      -> reporte_calidad()

  1. Normalizar columnas      -> eliminar_columnas() / limpiar_columnas() /
                                  renombrar_columnas()

  2. Corregir tipos de datos  -> reemplazar_caracter() / convertir_fecha() /
                                  convertir_int() / convertir_float() /
                                  convertir_datetime() / convertir_category() /
                                  convertir_bool()

  3. Tratar valores nulos     -> buscar_nulos() / columnas_con_nulos() /
                                  estadisticas_nulos() / imputar_media() /
                                  imputar_mediana() / imputar_moda() /
                                  imputar_constante() / eliminar_filas_nulas() /
                                  eliminar_columnas_nulas()

  3.5 Tratar columnas binarias -> preparar_binaria_ml() / crear_binaria_ml()

  4. Eliminar duplicados      -> buscar_duplicados() / eliminar_duplicados()

  5. Limpiar texto inconsistente -> ver_valores_string() / limpiar_texto() /
                                  reemplazar_texto() / reemplazar_texto_dic() /
                                  eliminar_acentos()

  6. Eliminar variables irrelevantes -> buscar_constantes() /
                                  buscar_columnas_vacias() / buscar_ids() /
                                  buscar_alta_cardinalidad()

  6.5 Validar coherencia entre columnas (reglas de negocio)
        -> NO tiene función propia en este módulo: las reglas dependen
        de cada dataset (ej. "si contacto_previo='no_contactado',
        dias_contacto debe ser 999"; "fecha_alta no puede ser posterior
        a fecha"). Ver la plantilla general (comprobaciones vectorizadas
        + value_counts() por regla) en sp_utils.py paso 19, sección "Reglas de
        negocio / coherencia entre columnas".

  7. Detectar y tratar outliers -> detectar_outliers_todas() /
                                  detectar_outliers_iqr() /
                                  eliminar_outliers_iqr() /
                                  winsorizar_columnas()

  8. Validación final y guardado -> reporte_calidad() (de nuevo, para comparar
                                  antes/después) / guardar_csv()

Este módulo NO hace feature engineering avanzado (escalado, encoding,
variables polinómicas...) — eso vive en el soporte de Modelado, que se
usa más adelante, cuando el dataset ya está limpio.

Extra (no numerado, utilidad puntual): extraer_fecha() para descomponer
una columna datetime en año/mes/día/trimestre/día de la semana.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


# ============================================================================
# 0. CALIDAD DEL DATASET
# ============================================================================



def reporte_calidad(df, n_ejemplos=3):
    """
    Proporciona una visión rápida del contenido del DataFrame.

    Muestra el tipo de dato de cada columna y varios ejemplos de
    valores reales para facilitar la inspección inicial del dataset.

    Parameters
    ----------
    df : DataFrame

    n_ejemplos : int, default=3
        Número de ejemplos únicos a mostrar por columna.

    Returns
    -------
    DataFrame
        Resumen con:
        - tipo: tipo de dato de la columna.
        - ejemplos: primeros valores únicos encontrados.

    """
    reporte = pd.DataFrame({
        "tipo": df.dtypes.astype(str)
    })

    reporte["ejemplos"] = [
        list(
            df[col]
            .dropna()
            .astype(str)
            .unique()[:n_ejemplos]
        )
        for col in df.columns
    ]

    return reporte

# ============================================================================
# 1. NORMALIZAR COLUMNAS
# ============================================================================

def eliminar_columnas(df, columnas):
    """Elimina las columnas indicadas."""
    return df.drop(columns=columnas)


def limpiar_columnas(df):
    """Normaliza los nombres de columna a snake_case."""
    nuevo = df.copy()
    columnas = []

    for col in nuevo.columns:
        col_limpia = col.strip().lower()
        col_limpia = col_limpia.replace(' ', '_')
        columnas.append(col_limpia)

    nuevo.columns = columnas
    return nuevo


def renombrar_columnas(df, diccionario):
    """Renombra columnas a partir de un diccionario {nombre_actual: nombre_nuevo}."""
    nuevo = df.rename(columns=diccionario)
    return nuevo

    # imputar --> df_l.columns.tolist() para ver los cambios

# ============================================================================
# 2. CORREGIR TIPOS DE DATOS
# ============================================================================

def reemplazar_caracter(df, columnas, buscar, reemplazo):
    """
    Sustituye un carácter (o fragmento de texto) DENTRO de los valores
    de una o varias columnas. A diferencia de reemplazar_texto(), que
    compara el valor completo de la celda, esta función busca y
    sustituye subcadenas — útil por ejemplo para cambiar comas
    decimales por puntos antes de convertir a numérico.

    Ejemplo
    -------
    reemplazar_caracter(df, ['euribor_3m', 'ipc', 'conf_consumidor', 'empleados'], ',', '.')
    """

    nuevo = df.copy()
    for col in columnas:
        nuevo[col] = nuevo[col].astype(str).str.replace(buscar, reemplazo, regex=False)
    return nuevo

def convertir_fecha(df, col_fecha):
    """Convierte una columna de texto con fechas en español a datetime."""
    meses = {
        'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
        'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
        'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
    }

    nuevo = df.copy()
    for mes, num in meses.items():
        nuevo[col_fecha] = nuevo[col_fecha].str.replace(f'-{mes}-', f'-{num}-', regex=False)

    nuevo[col_fecha] = pd.to_datetime(nuevo[col_fecha], format='%d-%m-%Y', errors='coerce')
    return nuevo


def convertir_int(df, col_int):
    """
    Convierte columnas a número entero. Los valores que no se puedan
    convertir se transforman en nulos (NaN) en vez de dar error
    (equivalente a errors='coerce').

    Parameters
    ----------
    df : DataFrame
    columnas : list[str]

    Returns
    -------
    DataFrame con las columnas convertidas.
    """
    nuevo = df.copy()

    if isinstance(col_int, str):
        col_int = [col_int]

    for col in col_int:
        nuevo[col] = pd.to_numeric(nuevo[col], errors='coerce').astype('Int64')
    return nuevo


    
def convertir_float(df, col_float):
    """Convierte una o varias columnas a número decimal."""
    nuevo = df.copy()
    if isinstance(col_float, str):
        col_float = [col_float]
    for col in col_float:
        nuevo[col] = pd.to_numeric(nuevo[col], errors='coerce')
    return nuevo


def convertir_datetime(df, col_date, formato=None, unidad='s'):
    """Convierte una o varias columnas a fecha, forzando siempre
    la misma resolución (por defecto segundos: datetime64[s]).
    
    Formato explícito si pandas no lo detecta bien (ej. '%d/%m/%Y').
    unidad: 's', 'ms', 'us' o 'ns' según la precisión que necesites.
    """
    nuevo = df.copy()
    if isinstance(col_date, str):
        col_date = [col_date]
    for col in col_date:
        nuevo[col] = pd.to_datetime(nuevo[col], format=formato, errors='coerce').astype(f'datetime64[{unidad}]')
    return nuevo

def convertir_category(df, col_cat):
    """Convierte una o varias columnas a tipo category."""
    nuevo = df.copy()
    if isinstance(col_cat, str):
        col_cat = [col_cat]
    for col in col_cat:
        nuevo[col] = nuevo[col].astype('category')
    return nuevo


def convertir_bool(df, col_bool, mapeo=None):
    """Convierte una o varias columnas a booleano (True/False)."""
    nuevo = df.copy()
    if isinstance(col_bool, str):
        col_bool = [col_bool]
    for col in col_bool:
        if mapeo:
            nuevo[col] = nuevo[col].map(mapeo)
        nuevo[col] = nuevo[col].astype('boolean')
    return nuevo


# ============================================================================
# 3. TRATAR VALORES NULOS
# ============================================================================

def buscar_nulos(df):
    """Columnas con nulos: cantidad y % de cada una."""
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    porcentaje = round(nulos / len(df) * 100, 2)
    return pd.DataFrame({'nulos': nulos, '%_nulos': porcentaje}).sort_values(
            '%_nulos',
            ascending=False
        )

def columnas_con_nulos(df):
    """Lista de nombres de columnas que tienen al menos un nulo."""
    nulos = df.isna().sum()
    return list(nulos[nulos > 0].index)


def estadisticas_nulos(df, columnas):
    """Describe() de las columnas indicadas (las que decidas mirar tras buscar_nulos)."""
    print('=' * 70)
    print('RECORDATORIO PARA VALORAR LA IMPUTACIÓN')
    print('=' * 70)
    print('• Media ≈ mediana: la media podría ser razonable.')
    print('• Media muy distinta de mediana: valorar la mediana.')
    print('• Std muy alta: posible presencia de outliers.')
    print('• Variables categóricas: usar la moda.')
    print('=' * 70)
    return df[columnas].describe(include="all").T


def imputar_media(df, col_media):
    """Rellena los nulos con la media de cada columna. Si la columna
    es de tipo entero, redondea la media antes de imputar (evita el
    error de intentar meter un decimal en una columna Int64)."""
    nuevo = df.copy()
    if isinstance(col_media, str):
        col_media = [col_media]
    for col in col_media:
        valor = nuevo[col].mean()
        if 'int' in str(nuevo[col].dtype).lower():
            valor = round(valor)
        nuevo[col] = nuevo[col].fillna(valor)
    return nuevo


def imputar_mediana(df, col_mediana):
    """Rellena los nulos con la mediana de cada columna. Si la columna
    es de tipo entero, redondea la mediana antes de imputar (evita el
    error de intentar meter un decimal en una columna Int64)."""
    nuevo = df.copy()
    if isinstance(col_mediana, str):
        col_mediana = [col_mediana]
    for col in col_mediana:
        valor = nuevo[col].median()
        if 'int' in str(nuevo[col].dtype).lower():
            valor = round(valor)
        nuevo[col] = nuevo[col].fillna(valor)
    return nuevo


def imputar_moda(df, col_moda):
    """Rellena los nulos con la moda (valor más frecuente). Es la
    opción habitual para columnas categóricas."""
    nuevo = df.copy()

    if isinstance(col_moda, str):
        col_moda = [col_moda]

    for col in col_moda:
        moda = nuevo[col].mode()
        if moda.empty:
            print(f'⚠ Aviso: "{col}" no tiene valores válidos, no se puede calcular la moda. Se deja sin cambios.')
            continue
        nuevo[col] = nuevo[col].fillna(moda[0])
    return nuevo


def imputar_constante(df, columnas, valor):
    """Rellena los nulos con un valor fijo indicado (ej. 'desconocido', 0)."""
    nuevo = df.copy()
    nuevo[columnas] = nuevo[columnas].fillna(valor)
    return nuevo


def eliminar_filas_nulas(df):
    """Elimina las filas que tengan al menos un valor nulo."""
    return df.dropna()


def eliminar_columnas_nulas(df, umbral=1.0):
    """
    Elimina columnas cuyo porcentaje de nulos sea igual o mayor al
    umbral indicado.

    umbral=1.0 -> elimina solo columnas 100% nulas (vacías).
    umbral=0.5 -> elimina columnas con 50% o más de nulos.
    """
    porcentaje = df.isna().mean()
    columnas = porcentaje[porcentaje >= umbral].index
    return df.drop(columns=columnas)

# ============================================================================
# 3.5. TRATAR COLUMNAS BINARIAS 
# ============================================================================

def preparar_binaria_ml(df, columna, valor_imputacion=None):
    """
    Crea una versión "_ml" de una columna binaria (0/1) que contiene
    nulos, lista para modelado: imputa los nulos y fuerza el tipo a
    int estándar (sin nulos), dejando la columna original intacta
    para trazabilidad. ya están en formato binario pero tienen nulos.

    Pensada para columnas que ya son numéricas (0/1) pero tienen
    valores nulos (ej. tras convertir_int() sobre un "unknown" que
    se ha vuelto NaN) — no sirve para columnas de texto, para eso
    usa crear_binaria_ml().

    Parameters
    ----------
    df : DataFrame
    columna : str
        Nombre de la columna binaria (con nulos) a preparar.
    valor_imputacion : int | float | None, default None
        Valor con el que rellenar los nulos. Si es None, se usa la
        moda de la columna. Indícalo explícitamente (ej. 0) si
        prefieres decidir el valor tú mismo en vez de dejarlo a la
        moda.

    Returns
    -------
    DataFrame con una columna nueva "{columna}_ml", numérica y sin
    nulos. La columna original no se modifica.

    Avisa por consola
    ------------------
    - Nº y % de filas imputadas.
    - Si el % imputado supera el 10%, un aviso adicional recordando
      valorar si la ausencia del dato está relacionada con la
      variable objetivo antes de asumir la moda sin más.

    Ejemplo
    -------
    preparar_binaria_ml(df, 'impago')
    preparar_binaria_ml(df, 'hipoteca', valor_imputacion=0)
    """
    nuevo = df.copy()

    # Si se pasa una sola columna, la convertimos en lista
    if isinstance(columna, str):
        columnas = [columna]
    else:
        columnas = columna

    for col in columnas:

        if col not in nuevo.columns:
            raise KeyError(f'La columna "{col}" no existe en el DataFrame.')

        col_ml = f'{col}_ml'

        n_nulos = nuevo[col].isna().sum()
        porcentaje = n_nulos / len(nuevo) * 100

        if valor_imputacion is None:
            moda = nuevo[col].mode()

            if moda.empty:
                raise ValueError(
                    f'No se puede calcular la moda de "{col}" porque '
                    'la columna no tiene valores válidos.'
                )

            imputacion = moda.iloc[0]

        else:
            imputacion = valor_imputacion

        nuevo[col_ml] = (
            nuevo[col]
            .fillna(imputacion)
            .astype(int)
        )

        print(f'✓ Columna creada: "{col_ml}"')
        print(
            f'  Imputados: {n_nulos} '
            f'({porcentaje:.2f}%) con valor {imputacion}'
        )

        if porcentaje > 10:
            print(
                f'  ⚠ AVISO: {porcentaje:.1f}% imputado es un porcentaje alto.'
            )
            print(
                '    Valora si la ausencia del dato podría estar relacionada '
                'con la variable objetivo.'
            )

    return nuevo

def crear_binaria_ml(df, columna, mapeo):
    """
    Crea una versión "_ml" de una columna categórica de 2 valores,
    mapeándola a 0/1 según el diccionario indicado. Conserva la
    columna original intacta.

    Pensada para columnas de texto SIN nulos que solo necesitan
    traducirse a numérico (ej. 'objetivo': 'no'/'yes' -> 0/1) — si
    la columna además tiene nulos, usa preparar_binaria_ml() o
    combina ambos criterios según el caso.

    Para varias columnas antes
    for columna in columnas:
    df_l = sl.crear_binaria_ml(df_l, columna)

    Parameters
    ----------
    df : DataFrame
    columna : str
        Nombre de la columna categórica a mapear.
    mapeo : dict
        Diccionario {valor_original: valor_numerico}, ej.
        {'no': 0, 'yes': 1}.

    Returns
    -------
    DataFrame con una columna nueva "{columna}_ml". La columna
    original no se modifica.

    Avisa por consola
    ------------------
    - Si todos los valores se han mapeado correctamente.
    - Si aparece algún valor no incluido en el diccionario (de lo
      contrario se convertiría en NaN de forma silenciosa).

    Ejemplo
    -------
    crear_binaria_ml(df, 'objetivo', {'no': 0, 'yes': 1})
    """
    nuevo = df.copy()
    col_ml = f'{columna}_ml'

    nuevo[col_ml] = nuevo[columna].map(mapeo)

    no_mapeados = nuevo[nuevo[col_ml].isna() & nuevo[columna].notna()][columna].unique()
    if len(no_mapeados) > 0:
        print(f'⚠ Aviso: valores en "{columna}" no incluidos en el mapeo, se convirtieron en NaN: {list(no_mapeados)}')
    else:
        print(f'✓ Columna creada: "{col_ml}" — todos los valores mapeados correctamente.')

    return nuevo



# ============================================================================
# 4. ELIMINAR DUPLICADOS
# ============================================================================

def buscar_duplicados(df):
    """Devuelve cuántas filas están duplicadas por completo."""
    return int(df.duplicated().sum())


def eliminar_duplicados(df, keep='first'):
    """
    Elimina filas duplicadas.

    keep='first' -> conserva la primera aparición (por defecto).
    keep='last'  -> conserva la última aparición.
    keep=False   -> elimina TODAS las filas que tengan algún duplicado.
    """
    return df.drop_duplicates(keep=keep)


# ============================================================================
# 5. LIMPIAR TEXTO INCONSISTENTE
# ============================================================================

def ver_valores_string(df, excluir=None):
    """Muestra los valores únicos de las columnas de tipo texto."""
    if excluir is None:
        excluir = []
    elif isinstance(excluir, str):
        excluir = [excluir]

    columnas = df.select_dtypes(include='str').columns
    columnas = [col for col in columnas if col not in excluir]

    for col in columnas:
        print(f'\n--- {col} ---')
        print(df[col].unique())


def limpiar_texto(df, columnas):
    """Normaliza texto: quita espacios sobrantes, pasa a minúsculas
    y colapsa espacios múltiples en uno solo."""
    nuevo = df.copy()

    if isinstance(columnas, str):
        columnas = [columnas]

    for col in columnas:
        nuevo[col] = (
            nuevo[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r'\s+', ' ', regex=True)
            .str.replace(' ', '_', regex=False)
        )
    return nuevo



def reemplazar_texto(df, columna, buscar, reemplazo):
    """
    Reemplaza valores concretos dentro de una columna.

    Ejemplo
    -------
    reemplazar_texto(df, 'ciudad', ['Sevilla ', ' sevilla'], 'sevilla')
    """
    nuevo = df.copy()
    nuevo[columna] = nuevo[columna].replace(buscar, reemplazo)
    return nuevo

def reemplazar_texto_dic(df, columna, reemplazos):
    """
    Reemplaza valores concretos dentro de una columna usando un diccionario.

    Ejemplo
    -------
    reemplazar_texto(
        df,
        'ciudad',
        {
            'Sevilla ': 'sevilla',
            ' sevilla': 'sevilla',
            'MADRID': 'madrid'
        }
    )
    """
    nuevo = df.copy()
    nuevo[columna] = nuevo[columna].replace(reemplazos)
    return nuevo


def eliminar_acentos(df, columnas):
    """Elimina tildes/acentos de las columnas de texto indicadas."""
    nuevo = df.copy()

    if isinstance(columnas, str):
        columnas = [columnas]

    for col in columnas:
        nuevo[col] = nuevo[col].apply(
            lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode()
            if pd.notna(x) else x
        )
    return nuevo


# ============================================================================
# EXTRA — FECHAS (utilidad puntual, no ligada a un paso numerado)
# ============================================================================

def extraer_fecha(df, columna):
    """
    Descompone una columna datetime en año, mes, día, día de la
    semana y trimestre, como columnas nuevas.

    Requiere que la columna ya esté convertida a datetime
    (ver convertir_datetime()).
    """
    nuevo = df.copy()
    nuevo[f'{columna}_anio'] = nuevo[columna].dt.year
    nuevo[f'{columna}_mes'] = nuevo[columna].dt.month
    nuevo[f'{columna}_dia'] = nuevo[columna].dt.day
    nuevo[f'{columna}_dia_semana'] = nuevo[columna].dt.day_name()
    nuevo[f'{columna}_trimestre'] = nuevo[columna].dt.quarter
    return nuevo

# ============================================================================
# 6. ELIMINAR VARIABLES IRRELEVANTES
# ============================================================================

def buscar_constantes(df):
    """Devuelve las columnas que tienen un único valor (no aportan
    información, se pueden eliminar)."""
    return [c for c in df.columns if df[c].nunique(dropna=False) == 1]


def buscar_columnas_vacias(df):
    """Devuelve las columnas que están 100% vacías (todo nulos)."""
    return df.columns[df.isna().all()].tolist()


def buscar_ids(df):
    """
    Devuelve columnas donde cada fila tiene un valor distinto
    (posibles identificadores: order_id, id_cliente...). Útiles
    para relacionar tablas, pero normalmente no sirven como métrica
    ni como variable de agrupación.
    """
    return [c for c in df.columns if df[c].nunique() == len(df)]


def buscar_alta_cardinalidad(df, umbral=0.90):
    """
    Devuelve columnas donde el % de valores únicos supera el umbral
    (por defecto 90%) sin llegar a ser un ID puro. Candidatas a
    revisar antes de usarlas como variable categórica.
    """
    columnas = []
    for c in df.columns:
        if (df[c].nunique() / len(df)) >= umbral:
            columnas.append(c)
    return columnas



# ============================================================================
# 7. DETECTAR Y TRATAR OUTLIERS
# ============================================================================

def detectar_outliers_todas(df):
    """
    Recorre TODAS las columnas numéricas y devuelve un resumen (una
    fila por columna) de las que tienen outliers según el criterio IQR.

    Es la vista rápida de Fase 2: úsala para decidir qué columnas
    merece la pena mirar de cerca, y luego usa detectar_outliers_iqr()
    sobre esa columna en concreto si necesitas el detalle completo
    (Q1, Q3, describe de los outliers...).

    Returns
    -------
    DataFrame con columnas: columna, outliers, %_outliers,
    limite_inferior, limite_superior. Ordenado de más a menos outliers.
    """
    filas = []
    for columna in df.select_dtypes(include='number').columns:
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr

        n_outliers = ((df[columna] < inferior) | (df[columna] > superior)).sum()

        if n_outliers > 0:
            filas.append({
                'columna': columna,
                'outliers': n_outliers,
                '%_outliers': round(n_outliers / len(df) * 100, 2),
                'limite_inferior': round(inferior, 3),
                'limite_superior': round(superior, 3),
            })

    resumen = pd.DataFrame(filas).sort_values('outliers', ascending=False).reset_index(drop=True)
    return resumen


def detectar_outliers_iqr(df, columna):
    """
    Devuelve las filas consideradas outlier en una columna, usando
    el criterio del rango intercuartílico (IQR): fuera de
    [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Parameters
    ----------
    df : DataFrame
    columna : str

    Returns
    -------
    DataFrame con solo las filas outlier de esa columna.
    """
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    iqr = q3 - q1
    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr

    outliers = df[(df[columna] < inferior) | (df[columna] > superior)]

    print(f'Columna: {columna}')
    print(f'Límite inferior: {inferior:.3f}  |  Límite superior: {superior:.3f}')
    print(f'Outliers encontrados: {len(outliers)} ({len(outliers) / len(df) * 100:.2f}%)')

    return outliers

# outliers_campana = detectar_outliers_iqr(df, 'campana')
# outliers_campana['campana'].describe()          # estadísticos solo de esos outliers
# outliers_campana.to_csv('revisar_campana.csv')  # exportar para revisión manual
# outliers_campana['campana'].sort_values(ascending=False).head(5)  # los 5 más extremos


def eliminar_outliers_iqr(df, columna, umbral_aviso=10.0):
    """
    Elimina del DataFrame las filas consideradas outlier en una
    columna (criterio IQR). A diferencia de capar_outliers_iqr(),
    que solo recorta el valor, esta función BORRA la fila completa
    — es una decisión más agresiva, ya que se pierde esa información.

    Avisa si el % de outliers detectado supera umbral_aviso, porque
    un porcentaje muy alto suele indicar que NO son errores de
    captura, sino valores extremos reales (la variable tiene una
    distribución con cola larga) — en ese caso, suele ser mejor usar
    capar_outliers_iqr() en vez de eliminar filas.

    Parameters
    ----------
    df : DataFrame
    columna : str
    umbral_aviso : float
        % de outliers a partir del cual se muestra el aviso
        (por defecto 10.0).

    Returns
    -------
    DataFrame sin las filas outlier de esa columna.
    """
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    iqr = q3 - q1
    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr

    es_outlier = (df[columna] < inferior) | (df[columna] > superior)
    n_outliers = int(es_outlier.sum())
    porcentaje = n_outliers / len(df) * 100

    print(f'Columna: {columna}')
    print(f'Límite inferior: {inferior:.3f}  |  Límite superior: {superior:.3f}')
    print(f'Filas a eliminar: {n_outliers} ({porcentaje:.2f}%)')

    if porcentaje > umbral_aviso:
        print('-' * 70)
        print(f'⚠ AVISO: el porcentaje de outliers supera el {umbral_aviso}%.')
        print('  Esto suele indicar que NO son errores de captura, sino valores')
        print('  extremos reales (distribución con cola larga). Antes de eliminar,')
        print('  valora usar capar_outliers_iqr() para no perder tanta información.')
        print('-' * 70)

    return df[~es_outlier]

def winsorizar_columnas(df, columnas, sufijo='_wz'):
    """
    'Winsoriza' una o varias columnas: en vez de eliminar los outliers,
    los recorta (clip) a los límites del IQR. Crea una columna nueva
    con el sufijo indicado (por defecto '_wz'), conservando la
    original intacta para reportar cifras reales de negocio.

    Ejemplo
    -------
    capar_outliers_iqr(df, ['campana', 'duracion'])
    # crea 'campana_wz' y 'duracion_wz'
    """
    nuevo = df.copy()

    if isinstance(columnas, str):
        columnas = [columnas]

    for col in columnas:
        q1 = nuevo[col].quantile(0.25)
        q3 = nuevo[col].quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr

        nuevo[f'{col}{sufijo}'] = nuevo[col].clip(inferior, superior)

    return nuevo



# ============================================================================
# 8. VALIDACIÓN FINAL Y GUARDADO
# ============================================================================

def guardar_csv(df, ruta):
    """
    Guarda el DataFrame en csv, listo para usarse en Python, Power BI
    o Excel sin problemas de codificación.

    Usa encoding='utf-8-sig' (UTF-8 con marca BOM): necesario para que
    programas como Power BI o Excel interpreten bien tildes y eñes.
    Sin esto, "sevilla" puede leerse mal fuera de Python.

    Parameters
    ----------
    df : DataFrame
    ruta : str
        Ruta de destino, ej. '../data/processed/02_datos_limpios.csv'
    """
    df.to_csv(ruta, index=False, encoding='utf-8-sig')

    print('=' * 70)
    print('ARCHIVO GUARDADO')
    print('=' * 70)
    print(f'Ruta:     {ruta}')
    print(f'Filas:    {df.shape[0]}')
    print(f'Columnas: {df.shape[1]}')
    print('=' * 70)