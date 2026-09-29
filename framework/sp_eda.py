"""
sp_eda.py

===================

Caja de herramientas de Fase 3 — Análisis Descriptivo y Visualización (EDA).

Sigue el flujo: Pre-EDA -> Univariante -> Bivariante -> Temporal -> Global.

    exploracion()          -> vista rápida: sample, info, describe, nulos y duplicados

    eda_numericas()        -> EDA de variables numéricas: estadísticas descriptivas,
                              distribuciones y detección visual de outliers
                            
    eda_categoricas()      -> EDA de variables categóricas: frecuencias,
                            cardinalidad y distribución de categorías

    eda_datetime()         -> EDA de variables temporales: tipos, rangos,
                              frecuencias y características de fechas

    columnas_numericas()   -> Obtiene las columnas numéricas y muestra sus 
                              estadísticas descriptivas  

    columnas_datetime()    -> Obtiene las columnas datetime y muestra sus 
                              estadísticas descriptivas       
                              
    corr_objetivo()        -> análisis de correlación de las variables numéricas
                              respecto a una variable objetivo
                              
    columnas_categoricas() -> Obtiene las columnas categóricas y muestra sus 
                              estadísticas descriptivas

    chi2_objetivo()        -> análisis de la relacion entre variables categoricas 
                              respecto a una variable objetivo                                            

    scatterplot()          -> scatterplot entre DOS columnas numéricas

    barplot()              -> barplot de una métrica numérica agrupada
                              por una variable categórica

    boxplot_bivar()        -> boxplot de una variable numérica según
                              una variable categórica

    countplot_hue()        -> countplot de una variable categórica,
                              desglosada mediante hue

    lineplot()             -> lineplot temporal de una métrica numérica,
                              resampleada y agregada según la frecuencia indicada

    barplot_serie()         -> gráfico de una Serie de pandas:
                               bar, barh o line                              

    matriz_correlacion()   -> heatmap de correlaciones + tabla de pares
                              más correlacionados                               

    exploracion_grupos()   -> Compara métricas numéricas entre grupos definidos                          
                              
    groupby()              -> Agrupa una variable numérica utilizando una o varias
                                  variables categóricas y aplica una función de agregación.
    
    crosstab()             -> Genera una tabla de contingencia (crosstab) entre una
                                  variable categórica y una o varias variables categóricas.
 
    countplot()            -> countplot de UNA columna categórica

    recuento_valores()     -> recuento y resumen de los valores de UNA columna 
                              con variable discreta

    histplot()             -> histograma de UNA columna numérica

    boxplot()              -> boxplot de UNA columna numérica (Outliers)

    dateplot()             -> representación temporal de UNA columna de fecha,
                              agrupada según la frecuencia indicada


Algunas Operaciones que no debemos olvidar
    VARIABLES RELACIONALES
    REGLAS DE NEGOCIO (Preferentemente se realiza en Fase de limpieza)
    PREGUNTAS DE NEGOCIO                    
"""
# ============================================================================
### 0️. IMPORTAR LIBRERIAS
# ============================================================================

# Tratamiento de Datos
import pandas as pd
import numpy as np
from IPython.display import display

# Visualizaciones
import seaborn as sns
import matplotlib.pyplot as plt

# Abtest (chi2)
import scipy.stats as stats

# Para que se muestren todas las columnas al inspeccionar los DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# ============================================================================
### 1️. CARGA DATAFRAME 
# ============================================================================

# sc.leer_csv(ruta, parse_dates=None, **kwargs)

# TASA DE CONVERSION GLOBAL
# df_l[" "].value_counts(normalize=True).mul(100).round(2)
# conv_global = (
    # df_l[" "]
    # .mean()
    # * 100 )
# print(f"Conversión global: {conv_global:.2f}%")


# ============================================================================
### 2️. EXPLORACION INICIAL DATASET 
# ============================================================================

def exploracion(df,cols_excluir=None, n=3):

 # Si no indicamos columnas a excluir,
    # creamos una lista vacia
    if cols_excluir is None:
        cols_excluir = []

    # Creamos un DataFrame sin las columnas excluidas
    df_exp = df.drop(columns=cols_excluir, errors="ignore")

    print('PRIMERAS COLUMNAS')
    display(df_exp.sample(n).T)
    print(":" * 100)
    print('INFORMACIÓN BÁSICA')
    display(df_exp.info())
    print(":" * 100)
    print('ESTADISTICOS')
    display(df_exp.describe(include="all").T)
    print(":" * 100)
    print('TAMAÑO DATAFRAME')
    print(df_exp.shape)
    print('DUPLICADOS')
    print(df_exp.duplicated().sum())
    print(":" * 100)
    print('NULOS')
    display(df_exp.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))
    print(":" * 100)
    print('PORCENTAJE DE NULOS')
    display((df_exp.isna().mean()[df.isna().mean() > 0] * 100).sort_values(ascending=False).round(2))
    print(":" * 100)
    print("REPRESENTACION GRAFICA DE VALORES NULOS")
    nulos = df_exp.isnull().sum()[df_exp.isnull().sum() > 0].sort_values(ascending=False)
    if nulos.empty:
        print('No hay columnas con nulos, no se genera el gráfico.')
    else:
        display(nulos.plot(kind='bar'))

# ============================================================================
### 3 GRUPOS TEMATICOS
# ============================================================================
##### 1. EDA BASICO (Visualizacion del Dataframe)
# ============================================================================

def eda_numericas(df, cols_excluir=None):
    """
    EDA de las columnas numéricas: estadísticos + histogramas + boxplots.
    """
    if cols_excluir is None:
        cols_excluir = []
    df_eda = df.drop(columns=cols_excluir, errors='ignore')
    num_cols = df_eda.select_dtypes(include='number').columns

    if len(num_cols) == 0:
        print('No hay columnas numéricas para analizar.')
        return

    print('VARIABLES NUMERICAS:', list(num_cols))
    print('=' * 100)
    print(df_eda[num_cols].describe().T.round(2))

    # Histogramas
    n_graficos = len(num_cols)
    ncols = 3
    nrows = (n_graficos + ncols - 1) // ncols

   
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, num_cols):
        sns.histplot(df_eda[col], bins=20, ax=ax)
        ax.set_title(col)
    for ax in axes[n_graficos:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

    
    # Boxplots
    n_graficos = len(num_cols)
    ncols = 1
    nrows = (n_graficos + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 2 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, num_cols):
        sns.boxplot(x=df_eda[col], ax=ax)
        ax.set_title(col)
    for ax in axes[n_graficos:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.show()

    return list(num_cols)


def eda_categoricas(df, cols_excluir=None):
    """
    EDA de las columnas categóricas: estadísticos + countplots 
    (se omite el gráfico de las columnas con más de 200 categorías).
    """
    if cols_excluir is None:
        cols_excluir = []
    df_eda = df.drop(columns=cols_excluir, errors='ignore')
    cat_cols = df_eda.select_dtypes(include=['string', 'category', 'object']).columns

    if len(cat_cols) == 0:
        print('No hay columnas categóricas para analizar.')
        return

    print('VARIABLES CATEGORICAS:', list(cat_cols))
    print('=' * 100)
    print(df_eda[cat_cols].describe().T)


    cat_cols_plot = [col for col in cat_cols if df_eda[col].nunique() <= 200]
    for col in cat_cols:
        if df_eda[col].nunique() > 200:
            print(f'Columna {col} tiene demasiadas categorías: {df_eda[col].nunique()}, no se grafica.')

    if len(cat_cols_plot) > 0:
        n_graficos = len(cat_cols_plot)
        ncols = 2
        nrows = (n_graficos + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, col in zip(axes, cat_cols_plot):
            sns.countplot(x=df_eda[col], order=df_eda[col].value_counts().index, ax=ax)
            ax.set_title(f'Distribución de {col}')
            ax.tick_params(axis='x', rotation=90)
        for ax in axes[n_graficos:]:
            ax.set_visible(False)
        plt.tight_layout()
        plt.show()

    return list(cat_cols)

def eda_datetime(df, cols_excluir=None):
    """
    EDA de las columnas de fecha: estadísticos + lineplot básico
    (nº de filas por mes, para ver la evolución en el tiempo).
    """
    if cols_excluir is None:
        cols_excluir = []
    df_eda = df.drop(columns=cols_excluir, errors='ignore')
    date_cols = df_eda.select_dtypes(include=['datetime', 'datetimetz']).columns

    if len(date_cols) == 0:
        print('No hay columnas de fecha para analizar.')
        return

    print('VARIABLES DATETIME:', list(date_cols))
    print('=' * 100)
    print(df_eda[date_cols].describe().T)

    fig, axes = plt.subplots(len(date_cols), 1, figsize=(10, 3.5 * len(date_cols)))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, date_cols):
        serie = df_eda[col].dropna().dt.to_period('M').value_counts().sort_index()
        serie.index = serie.index.to_timestamp()
        sns.lineplot(x=serie.index, y=serie.values, marker='o', ax=ax)
        ax.set_title(f'Evolución mensual de {col}')
        ax.set_ylabel('Nº de filas')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    return list(date_cols)
# ============================================================================
##### 2. ANALISIS POR TIPOS DE COLUMNAS
# ============================================================================

        # ============================================================================
        ##### COLUMNAS NUMÉRICAS
        # ============================================================================

def columnas_numericas(
    df,
    num_cols=None,
    cols_excluir=None
):
    """
    Obtiene las columnas numéricas de un DataFrame y muestra
    un resumen ejecutivo de cada variable.

    A diferencia de eda_numericas(), esta función NO genera
    gráficos.

    Muestra para cada variable:

    - media
    - mediana
    - coeficiente de variación (CV)
    - valor mínimo
    - valor máximo
    - rango normal según IQR
    - número de outliers
    - porcentaje de outliers

    Puede recibir una lista de columnas numéricas ya
    calculada (por ejemplo desde eda_numericas()) para
    evitar volver a detectarlas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    num_cols : list, default=None
        Lista de columnas numéricas a analizar.

        Si es None, se detectan automáticamente.

    cols_excluir : list, default=None
        Columnas que no se deben considerar.

    Returns
    -------
    None

    Ejemplos
    --------

    Detección automática:

    se.columnas_numericas(clientes)

    Reutilizando columnas obtenidas en eda_numericas():

    variables_clientes_num = se.eda_numericas(clientes)

    se.columnas_numericas(
        clientes,
        num_cols=variables_clientes_num
    )
    """

    if cols_excluir is None:
        cols_excluir = []

    df_eda = df.drop(
        columns=cols_excluir,
        errors="ignore"
    )

    # -------------------------
    # COLUMNAS NUMÉRICAS
    # -------------------------

    if num_cols is None:

        num_cols = (
            df_eda
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

    if not num_cols:
        print("No hay columnas numéricas para analizar.")
        return None

    print("VARIABLES NUMÉRICAS:", num_cols)
    print("=" * 100)

    for col in num_cols:

        serie = df_eda[col].dropna()

        q1 = serie.quantile(0.25)
        q3 = serie.quantile(0.75)

        iqr = q3 - q1

        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr

        outliers = serie[
            (serie < inferior) |
            (serie > superior)
        ]

        n_outliers = len(outliers)

        pct_outliers = (
            n_outliers / len(serie) * 100
        )

        media = serie.mean()

        cv = (
            serie.std() / media
            if media != 0
            else np.nan
        )

        print(f"\n{col}")
        print("-" * len(col))

        print(f"Media: {media:.2f}")
        print(f"Mediana: {serie.median():.2f}")

        if pd.notna(cv):
            print(f"CV: {cv:.2f}")
        else:
            print("CV: no definido (media = 0)")

        print(
            f"Mín-Máx: "
            f"{serie.min():.2f} - "
            f"{serie.max():.2f}"
        )

        print(
            f"Rango normal: "
            f"[{inferior:.2f}, {superior:.2f}]"
        )

        print(
            f"Outliers: "
            f"{n_outliers} "
            f"({pct_outliers:.2f}%)"
        )

    return None




    # ============================================================================
    ##### COLUMNAS CATEGÓRICAS
    # ============================================================================

def columnas_categoricas(
    df,
    cat_cols=None,
    cols_excluir=None,
    top_n=5
):
    """
    Obtiene las columnas categóricas de un DataFrame y muestra
    un resumen ejecutivo de cada variable.

    A diferencia de eda_categoricas(), esta función NO genera
    gráficos.

    Muestra para cada variable:

    - número de categorías
    - cardinalidad
    - porcentaje dominante
    - número de categorías con menos del 5%
    - tabla TOP N con frecuencia y porcentaje

    Definiciones
    ------------
    Cardinalidad:
        Número de categorías distintas.

        Baja  : < 10 categorías
        Media : 10 - 30 categorías
        Alta  : > 30 categorías

    Porcentaje dominante:
        Porcentaje que representa la categoría más frecuente.

    Categorías <5%:
        Número de categorías cuya frecuencia relativa
        es inferior al 5% del total.

    Parameters
    ----------
    df : pd.DataFrame

    cat_cols : list, default=None
        Lista de columnas categóricas a analizar.

        Si es None, se detectan automáticamente.

    cols_excluir : list, default=None
        Columnas que no se deben considerar.

    top_n : int, default=5
        Número de categorías a mostrar
        ordenadas por frecuencia.

    Returns
    -------
    None

    Ejemplos
    --------

    Detección automática:

    se.columnas_categoricas(clientes)

    Reutilizando columnas obtenidas con eda_categoricas():

    variables_clientes_cat = se.eda_categoricas(clientes)

    se.columnas_categoricas(
        clientes,
        cat_cols=variables_clientes_cat
    )
    """

    if cols_excluir is None:
        cols_excluir = []

    df_eda = df.drop(
        columns=cols_excluir,
        errors="ignore"
    )

    # --------------------------------------------------
    # COLUMNAS CATEGÓRICAS
    # --------------------------------------------------

    if cat_cols is None:

        cat_cols = (
            df_eda
            .select_dtypes(
                include=["object", "category", "string"]
            )
            .columns
            .tolist()
        )

    if not cat_cols:
        print("No hay columnas categóricas para analizar.")
        return None

    print("VARIABLES CATEGÓRICAS:", cat_cols)
    print("=" * 100)

    for col in cat_cols:

        frecuencias = (
            df_eda[col]
            .value_counts(dropna=False)
        )

        porcentajes = (
            df_eda[col]
            .value_counts(
                normalize=True,
                dropna=False
            )
            .mul(100)
            .round(2)
        )

        n_categorias = len(frecuencias)

        # -------------------------
        # CARDINALIDAD
        # -------------------------

        if n_categorias < 10:
            cardinalidad = "Baja"

        elif n_categorias <= 30:
            cardinalidad = "Media"

        else:
            cardinalidad = "Alta"

        porcentaje_dominante = porcentajes.iloc[0]

        categorias_raras = (
            porcentajes < 5
        ).sum()

        # -------------------------
        # TABLA RESUMEN
        # -------------------------

        tabla = pd.DataFrame({
            "unidades": frecuencias,
            "%": porcentajes
        })

        # -------------------------
        # IMPRESIÓN
        # -------------------------

        print(f"\n{col}")
        print("-" * len(col))

        print(f"Categorías: {n_categorias}")
        print(f"Cardinalidad: {cardinalidad}")

        print(
            f"Porcentaje dominante: "
            f"{porcentaje_dominante:.2f}%"
        )

        print(
            f"Categorías <5%: "
            f"{categorias_raras}"
        )

        print(f"\nTOP {top_n} CATEGORÍAS")
        print("." * 40)

        print(tabla.head(top_n))

        print("\n" + "=" * 100)   




    # ============================================================================
    ##### COLUMNAS DATETIME
    # ============================================================================
def columnas_datetime(
    df,
    date_cols=None,
    cols_excluir=None
):
    """
    Obtiene las columnas datetime de un DataFrame y muestra
    un resumen ejecutivo de cada variable temporal.

    A diferencia de eda_datetime(), esta función NO genera
    gráficos.

    Muestra para cada variable:

    - primer registro
    - último registro
    - rango temporal
    - número de registros
    - número de fechas únicas

    Puede recibir una lista de columnas datetime ya
    calculada (por ejemplo desde eda_datetime()) para
    evitar volver a detectarlas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.

    date_cols : list, default=None
        Lista de columnas datetime a analizar.

        Si es None, se detectan automáticamente.

    cols_excluir : list, default=None
        Columnas que no se deben considerar.

    Returns
    -------
    None

    Ejemplos
    --------

    Detección automática:

    se.columnas_datetime(df)

    Reutilizando columnas obtenidas con eda_datetime():

    variables_fecha = se.eda_datetime(df)

    se.columnas_datetime(
        df,
        date_cols=variables_fecha
    )
    """

    if cols_excluir is None:
        cols_excluir = []

    df_eda = df.drop(
        columns=cols_excluir,
        errors="ignore"
    )

    # --------------------------------------------------
    # COLUMNAS DATETIME
    # --------------------------------------------------

    if date_cols is None:

        date_cols = (
            df_eda
            .select_dtypes(
                include=["datetime", "datetimetz"]
            )
            .columns
            .tolist()
        )

    if not date_cols:
        print("No hay columnas datetime para analizar.")
        return None

    print("VARIABLES DATETIME:", date_cols)
    print("=" * 100)

    for col in date_cols:

        serie = df_eda[col].dropna()

        if len(serie) == 0:

            print(f"\n{col}")
            print("-" * len(col))
            print("La columna no contiene fechas válidas.")
            continue

        fecha_min = serie.min()
        fecha_max = serie.max()

        rango_dias = (
            fecha_max - fecha_min
        ).days

        print(f"\n{col}")
        print("-" * len(col))

        print(
            f"Primer registro: "
            f"{fecha_min}"
        )

        print(
            f"Último registro: "
            f"{fecha_max}"
        )

        print(
            f"Rango temporal: "
            f"{rango_dias} días"
        )

        print(
            f"Registros: "
            f"{len(serie)}"
        )

        print(
            f"Fechas únicas: "
            f"{serie.nunique()}"
        )

        print()

    return None


# ============================================================================
##### 3. ANALISIS DE RELACION CON VARIABLE OBJETIVO
# ============================================================================

def corr_objetivo(df, target, variables=None):
    """
    Calcula la correlación de todas las variables numéricas
    respecto a una variable objetivo binaria.

    Parameters
    ----------
    df : pd.DataFrame

    target : str o binaria
        Variable objetivo numérica (0/1).

    Returns
    -------
    pd.Series
        Correlaciones ordenadas por valor absoluto.
    """

    if variables is None:
        variables = df.select_dtypes(include="number").columns.tolist()

    variables = [v for v in variables if v != target]

    corr = (
        df[variables + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )

    print(f"Correlación con {target}:")
    print(corr.round(3))

    return corr

def chi2_objetivo(df, col_cat, tv):
    """
    Relación entre variables categóricas y la variable objetivo
    mediante Chi-cuadrado.
    Devuelve los p-values ordenados de menor a mayor
    (variables más relacionadas primero).

    Parameters
    ----------
    df : pd.DataFrame

    col_cat : str | list[str]
        Variable o lista de variables categóricas.

    tv : str
        Variable objetivo categórica.

    Returns
    -------
    pd.DataFrame
        Tabla con los p-values ordenados.
    """

    if isinstance(col_cat, str):
        col_cat = [col_cat]

    resultados = []

    for col in col_cat:

        tabla = pd.crosstab(df[col],df[tv])
        _, pvalue, _, _ = stats.chi2_contingency(tabla)

        resultados.append({"variable": col,"pvalue": round(pvalue, 5)})

    resultado = (
        pd.DataFrame(resultados)
        .sort_values("pvalue")
        .reset_index(drop=True)
        
    )
    print("* pvalue < 0.05 existe relacion con TV")
    print("-"*30)
    print(resultado)

    return resultado

# ============================================================================
### 4. EDA BIVARIANTE — (Analisis Bivariante)
# ============================================================================

    # --------------------------------------------------
    # SCATTERPLOTS num - num 
    # --------------------------------------------------

def scatterplot(df, col_x, col_y):
    """
    Relación entre dos variables numéricas.
    Dibuja un scatterplot y calcula la correlación de Pearson.
    """
    sns.scatterplot(data=df, x=col_x, y=col_y, alpha=0.35)
    plt.title(f"{col_x} vs {col_y}")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.tight_layout()
    plt.show()

    r = df[col_x].corr(df[col_y])
    print(f"Correlación de Pearson entre {col_x} y {col_y}: {r:.4f}")

    return None

    # --------------------------------------------------
    # BARPLOTS  cat - num (media)
    # --------------------------------------------------


def barplot(df, col_cat, col_num, estimator="mean", errorbar=None,figsize=(10,5)):
    """
    Estadístico (media, mediana, suma...) de una variable numérica
    según una variable categórica.
    Dibuja un barplot con ese estadístico.

    estimator: "mean", "median", "sum", etc.
    """

    plt.figure(figsize=figsize)

    sns.barplot(
        data=df,
        x=col_cat,
        y=col_num,
        estimator=estimator,
        errorbar=errorbar
    )

    plt.title(f"{estimator} de {col_num} por {col_cat}")
    plt.xlabel(col_cat)
    plt.ylabel(f"{estimator} de {col_num}")
    plt.xticks(rotation=20)

    plt.tight_layout()
    plt.show()

    valores = (
        df.groupby(col_cat)[col_num]
          .agg(estimator)
          .round(2)
          .sort_values(ascending=False)
    )

    print(f"\nTabla de {estimator} de {col_num} por {col_cat}:")
    print(valores)

    return None


    # --------------------------------------------------
    # BOXPLOT - BIVARIABLES cat-num COLUMNAS NUMERICAS
    # --------------------------------------------------

def boxplot_bivar(df, col_cat, col_num,figsize=(10,5)):
    """
    Distribución de una variable numérica según una variable categórica.
    Dibuja un boxplot (muestra mediana, dispersión y outliers).
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, y=col_cat, x=col_num)
    plt.title(f"Boxplot de {col_num} vs {col_cat}")
    plt.ylabel(col_cat)
    plt.xlabel(col_num)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    resumen = df.groupby(col_cat)[col_num].describe().round(2)
    print(f"Resumen de {col_num} por {col_cat}:")
    print(resumen)

    medias = df.groupby(col_cat)[col_num].mean()
    print("=" * 80)
    print(f"Diferencia de medias: {(medias.max() - medias.min()).round(2)}")

    return None

    # --------------------------------------------------
    # COUNTPLOT - HUE --- cat o num binarias + hue(cat) COLUMNAS CATEGORICAS
    # --------------------------------------------------


def countplot_hue(df, col_cat, hue, normalize=True, sort=True,figsize=(10,5)):
    """
    Relación entre dos variables categóricas.

    Dibuja un countplot de `col_cat` coloreado por `hue`.
    Muestra la tabla de contingencia de conteos y, opcionalmente,
    la tabla de porcentajes por fila.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables.

    col_cat : str
        Variable categórica representada en el eje x.

    hue : str
        Variable categórica utilizada para segmentar los conteos.

    normalize : bool, default=False
        Si es True, muestra también una tabla de porcentajes
        normalizada por filas.

    sort : bool, default=True
        Si es True, ordena las categorías de mayor a menor
        frecuencia.

    Returns
    -------
    None
    """

    # Orden de categorías
    orden = None
    if sort:
        orden = df[col_cat].value_counts().index

    # Gráfico
    plt.figure(figsize=figsize)

    sns.countplot(
        data=df,
        x=col_cat,
        hue=hue,
        order=orden
    )

    plt.title(f"Conteo de {col_cat} por {hue}")
    plt.xlabel(col_cat)
    plt.ylabel("Conteo")

    plt.xticks(rotation=90)

    plt.legend(
        title=hue,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

    # Tabla de conteos
    tabla_conteos = pd.crosstab(
        df[col_cat],
        df[hue]
    )

    if sort:
        tabla_conteos = tabla_conteos.loc[
            df[col_cat].value_counts().index
        ]

    print(f"\nTabla de conteos: {col_cat} vs {hue}")
    print(tabla_conteos)

    # Tabla de porcentajes
    if normalize:

        tabla_pct = pd.crosstab(
            df[col_cat],
            df[hue],
            normalize="index"
        ).round(3)

        if sort:
            tabla_pct = tabla_pct.loc[
                df[col_cat].value_counts().index
            ]

        print("\nTabla de porcentajes por fila:")
        print(tabla_pct)

    return None

    # --------------------------------------------------
    # LINEPLOT fecha-num
    # --------------------------------------------------

def lineplot(df, col_fecha, col_num, marker='o', freq="ME", agg="sum"):
    """
    Evolución de una métrica numérica a lo largo del tiempo.

    freq: "D" diario, "W" semanal, "ME" mensual, "YE" anual
    agg: "sum", "mean", "count", etc.
    """

    datos = df[[col_fecha, col_num]].dropna().copy()

    datos[col_fecha] = pd.to_datetime(datos[col_fecha])

    datos = datos.set_index(col_fecha)

    serie = datos[col_num].resample(freq).agg(agg)

    datos_plot = serie.reset_index()

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        data=datos_plot,
        x=col_fecha,
        y=col_num,
        marker=marker
    )

    plt.title(f"Evolución de {col_num} ({agg}, freq={freq})")
    plt.xlabel("Fecha")
    plt.ylabel(col_num)
    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()

    if not serie.empty:
        print(f"Media: {serie.mean():.2f}")
        print(f"Std: {serie.std():.2f}")
        print(f"Mínimo: {serie.idxmin()} -> {serie.min():.2f}")
        print(f"Máximo: {serie.idxmax()} -> {serie.max():.2f}")

    return None


    # --------------------------------------------------
    # BARPLOTS - SERIES
    # --------------------------------------------------

def barplot_serie(serie, titulo, kind="bar", color=None, xlabel=None, ylabel=None):
    """ Representa una Serie ya agregada (resultado de un groupby, value_counts,
    o cualquier cálculo previo), sin necesidad de volver a agregar datos.
    value_counts(), groupby().mean(),groupby().sum(),groupby().median(),crosstab()[columna]
    Útil para los casos en los que ya tienes el resultado calculado
    (ej. ventas por mes, tasa de devolución por canal) y solo
    necesitas graficarlo.

    Parameters
    ----------
    serie : pd.Series
        Serie ya agregada (el índice será el eje x, o el eje y si kind="barh").

    titulo : str
        Título del gráfico.

    kind : str
        Tipo de gráfico: "bar", "barh" o "line". Por defecto "bar".

    color : str
        Color de las barras/línea (opcional).

    xlabel : str
        Etiqueta del eje x (opcional, si no se indica usa el nombre del índice).

    ylabel : str
        Etiqueta del eje y (opcional, si no se indica usa el nombre de la serie)."""

    plt.figure(figsize=(8, 4))

    if kind == "bar":
        serie.plot(kind="bar", color=color)
        plt.xticks(rotation=30)

    elif kind == "barh":
        serie.plot(kind="barh", color=color)

    elif kind == "line":
        serie.plot(kind="line", marker="o", color=color)
        plt.xticks(rotation=30)

    else:
        raise ValueError('kind debe ser "bar", "barh" o "line"')

    plt.title(titulo)
    plt.xlabel(xlabel if xlabel else (serie.index.name or ""))
    plt.ylabel(ylabel if ylabel else (serie.name or ""))

    plt.tight_layout()
    plt.show()

    print(serie.round(2))

    return None

# --------------------------------------------------
### 5. EDA GLOBAL (Correlacion del dataframe)
# --------------------------------------------------

def matriz_correlacion(df, lista_cols_num=None, cols_excluir=None, top_n=10):
    """Calcula y representa la matriz de correlación
    y muestra los pares de variables más correlacionados."""

    if cols_excluir is None:
        cols_excluir = []

    # Seleccionar columnas numéricas
    if lista_cols_num is None:
        corr = df.corr(numeric_only=True)
    else:
        corr = df[lista_cols_num].corr()

    # Eliminar filas y columnas excluidas
    corr = corr.drop(
        index=cols_excluir,
        columns=cols_excluir,
        errors="ignore"
    )

    # Crear la figura
    plt.figure(
        figsize=(
            0.7 * len(corr.columns) + 2,
            0.6 * len(corr.columns) + 2
        )
    )

    # Crear máscara triangular
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Crear heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        annot_kws={"size": 7}
    )

    plt.tight_layout()
    plt.show()

    # Obtener pares de variables más correlacionados
    pares = corr.abs().unstack().sort_values(ascending=False)

    # Eliminar correlaciones de una variable consigo misma
    pares = pares[pares < 0.999].drop_duplicates()

    print(f"Top {top_n} pares con mayor correlación (valor absoluto):")
    print(pares.head(top_n))

    return corr

    # --------------------------------------------------
    ###  COMPARACIÓN DE GRUPOS
    # --------------------------------------------------

def exploracion_grupos(df, col_control, metricas):
    """
    Compara métricas numéricas entre los grupos definidos por una columna
    de control. Devuelve una tabla resumida mucho más útil para el análisis
    inferencial que un describe() completo.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame a analizar.

    col_control : str
        Columna categórica que define los grupos
        (ej: sexo, embarque, acompanado, titulo).

    metricas : str o list
        Métrica o lista de métricas numéricas a comparar
        (ej: sobrevivio_ml, edad, tarifa, familia).

    Returns
    -------
    pandas.DataFrame

    Para 2 grupos:
        métrica | grupo_1 | grupo_2 | diferencia

    La diferencia se calcula como:
        grupo_2 - grupo_1

    Para más de 2 grupos:
        métrica | grupo_1 | grupo_2 | grupo_3 | ...

    Examples
    --------
    exploracion_grupos(
        titanic,
        "sexo",
        ["sobrevivio_ml", "edad", "tarifa", "familia"]
    )

    exploracion_grupos(
        titanic,
        "embarque",
        "sobrevivio_ml"
    )
    """
    if isinstance(metricas, str):
        metricas = [metricas]

    grupos = list(df[col_control].unique())

    filas = []

    for metrica in metricas:

        fila = {"metrica": metrica}

        for grupo in grupos:

            valor = df.loc[
                df[col_control] == grupo,
                metrica
            ].mean()

            fila[grupo] = round(valor, 2)

        if len(grupos) == 2:

            fila["diferencia"] = round(
                fila[grupos[1]] - fila[grupos[0]],
                2
            )

        else:

            valores = [fila[g] for g in grupos]

            fila["rango"] = round(
                max(valores) - min(valores),
                2
            )

        filas.append(fila)

    resultado = pd.DataFrame(filas)

    if len(grupos) > 2:
        resultado = resultado.sort_values(
            "rango",
            ascending=False
        )

    display(resultado)

    return None



# ============================================================================
#  REGLAS DE NEGOCIO AGRUPACIONES GROUPBY . CROSSTAB
# ============================================================================

def groupby(df,col_num,col_cat,agg="mean",round_n=3,sort=False):
    """
    Variable numérica agrupada por categoria.
    Agrupa una variable numérica utilizando una o varias
    variables categóricas y aplica una función de agregación.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de trabajo.

    col_num : str
        Variable numérica a resumir.

    col_cat o tv : str | list[str]
        Variable o lista de variables categóricas utilizadas
        para agrupar.

    agg : str, default='mean'
        Estadístico a calcular.

        Ejemplos: - 'mean'-'median'-'sum'-'count'-'min'-'max'

    round_n : int, default=2
        Número de decimales a mostrar.

    sort : bool, default=False
        Si es True ordena el resultado de forma descendente.

    Returns
    -------
    pd.Series | pd.DataFrame
        Resultado de la agrupación.

    Notas
    -----
    Si la variable es binaria (0/1), 
    usar agg='mean' equivale a obtener la proporción o tasa media.

    """

    if isinstance(col_cat, str):
        col_cat = [col_cat]

    resultado = (
        df.groupby(col_cat)[col_num]
        .agg(agg)
        .round(round_n)
    )

    if len(col_cat) > 1: 
        resultado = resultado.unstack()

    if sort:
        resultado = resultado.sort_values(
            ascending=False
        )

  
    return resultado

def crosstab(df,col_cat,vars_cat,normalize="index",round_n=3,sort=False):
    """
    Tabla de contingencia entre variables categóricas.
    Genera una tabla de contingencia (crosstab) entre una
    variable categórica y una o varias variables categóricas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de trabajo.

    col_cat : str
        Variable categórica principal que aparecerá en las filas de la tabla.

    vars_cat : str | list[str]
        Variable o lista de variables categóricas que aparecerán en las columnas de la tabla.

    normalize : str | bool, default='index'
        Tipo de normalización.

        Opciones:
        - False: conteos absolutos.
        - 'index': porcentajes por fila.
        - 'columns': porcentajes por columna.
        - 'all': porcentajes globales.

    round_n : int, default=3
        Número de decimales.

    sort : bool, default=False
        Si es True ordena las filas por frecuencia de aparición de la variable principal.

    Returns
    -------
    pd.DataFrame
        Tabla de contingencia.

    Notas
    -----
    normalize='index', 
    suele ser la opción más útil para interpretar porcentajes de conversión y comparar categorías entre sí.
    """

    tabla = pd.crosstab(
        df[col_cat], [df[col] for col in [vars_cat] if isinstance(vars_cat, str)] if isinstance(vars_cat, str)
        else [df[col] for col in vars_cat],normalize=normalize)

    if normalize:
        tabla = tabla.round(round_n)

    if sort:
        orden = df[col_cat].value_counts().index
        tabla = tabla.loc[orden]


    return tabla


# ============================================================================
###  EXTRA GRAFICAS DE APOYO (Análisis univariante)
# ============================================================================

    # --------------------------------------------------
    # COUNTPLOT cat o num binarias
    # --------------------------------------------------

def countplot(df, col):
    """
    Distribución de una variable categórica.
    Dibuja un countplot ordenado por frecuencia.
    """
    order = df[col].value_counts().index

    sns.countplot(data=df, x=col, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # HISTPLOT num
    # --------------------------------------------------
def recuento_valores(df, col,n=5):
    """
    Frecuencias absolutas y relativas de una variable discreta.
    """
    print("\nVALORES")
    print(df[col].value_counts().head(n))

    print("\nPORCENTAJES")
    print(df[col].value_counts(normalize=True).mul(100).round(2).head(n))

def histplot(df, col, bins=30):
    """
    Distribución de una variable numérica.
    Dibuja un histograma.
    """

    sns.histplot(data=df, x=col, bins=bins)
    plt.title(f"Distribución de {col}")
    plt.xlabel(col)
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # BOXPLOT + DETECCIÓN DE OUTLIERS num (Outliers)
    # --------------------------------------------------

def boxplot(df, col,figsize=(10,2)):
    """
    Distribución de una variable numérica.
    Dibuja un boxplot y detecta outliers mediante el criterio IQR.

    """

        # -------------------------
        # BOXPLOT
        # -------------------------
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()



    # --------------------------------------------------
    # DATEPLOT  datetime
    # --------------------------------------------------

def dateplot(df, col_fecha, freq="ME"):
    """
    Distribución temporal de una variable datetime.
    freq: "D" diario, "W" semanal, "ME" mensual, "YE" anual
    """

    serie = (
        df.set_index(col_fecha)
          .resample(freq)
          .size()
    )

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        x=serie.index,
        y=serie.values,
        marker="o"
    )

    plt.title(f"Registros por {freq}")
    plt.xlabel("Fecha")
    plt.ylabel("Frecuencia")

    plt.show()
