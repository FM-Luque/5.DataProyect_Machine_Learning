"""
sp_eda.py

===================

Caja de herramientas de Fase 3 — Análisis Descriptivo y Visualización (EDA).

Sigue el flujo: Pre-EDA -> Univariante -> Bivariante -> Temporal -> Global.

Solo contenido descriptivo (nada de tests de hipótesis / p-valores:
eso vive en sp_abtest_completo.py, Fase 4).

    exploracion()          -> vista rápida: sample, info, describe, nulos y duplicados

    eda_consultas()        -> EDA de variables de consulta: tipos, frecuencias,
                            valores únicos y principales características

    eda_numericas()        -> EDA de variables numéricas: estadísticas descriptivas,
                            distribuciones y detección visual de outliers

    eda_categoricas()      -> EDA de variables categóricas: frecuencias,
                            cardinalidad y distribución de categorías

    eda_datetime()         -> EDA de variables temporales: tipos, rangos,
                            frecuencias y características de fechas

    countplot()            -> countplot de UNA columna categórica

    recuento_valores()     -> recuento y resumen de los valores de UNA columna

    histplot()             -> histograma de UNA columna numérica

    boxplot()              -> boxplot de UNA columna numérica

    dateplot()             -> representación temporal de UNA columna de fecha,
                            agrupada según la frecuencia indicada

    scatterplot()          -> scatterplot entre DOS columnas numéricas

    barplot()              -> barplot de una métrica numérica agrupada
                            por una variable categórica

    boxplot_bivar()        -> boxplot de una variable numérica según
                            una variable categórica

    countplot_hue()        -> countplot de una variable categórica,
                            desglosada mediante hue

    lineplot()              -> lineplot temporal de una métrica numérica,
                            resampleada y agregada según la frecuencia indicada

    barplot_serie()         -> gráfico de una Serie de pandas:
                            bar, barh o line

matriz_correlacion()   -> heatmap de correlaciones + tabla de pares
                        más correlacionados

corr_objetivo()        -> análisis de correlación de las variables numéricas
                        respecto a una variable objetivo

Todas asumen que le pasas el DataFrame ya al nivel correcto
(df = nivel pedido / order_id, clientes = nivel cliente / id_cliente).

Repasa la Guía de claves de Fase 3 antes de decidir qué tabla usar.

"""

# Tratamiento de Datos
import pandas as pd
import numpy as np
from IPython.display import display

# Visualizaciones
import seaborn as sns
import matplotlib.pyplot as plt


# Para que se muestren todas las columnas al inspeccionar los DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# ============================================================================
# 0. CARGA
# ============================================================================

# sc.leer_csv(ruta, parse_dates=None, **kwargs)


# ============================================================================
# 1. EDA PRELIMINAR - CALIDAD DEL DATASET
# ============================================================================

    # --------------------------------------------------
    # EXPLORACION DEL DATA FRAME
    # --------------------------------------------------

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

    # --------------------------------------------------
    # EDA PREMILINAR DEL DATA FRAME
    # --------------------------------------------------
def eda_consultas(df, cols_excluir=None):
    """
    Funcion que proporciona un EDA rapido.Realizado en consultas_referencias.

    Parameters
    ----------
    df : DataFrame
        DataFrame que queremos analizar.

    n : int
        Numero de decimales para las estadisticas numericas.

    cols_excluir : list
        Lista de columnas que no queremos analizar.
    """

    # Si no indicamos columnas a excluir,
    # creamos una lista vacia
    if cols_excluir is None:
        cols_excluir = []

    # Creamos un DataFrame sin las columnas excluidas
    df_eda = df.drop(columns=cols_excluir, errors="ignore")

    # --------------------------------------------------
    # IDENTIFICACION DE TIPOS DE COLUMNAS
    # --------------------------------------------------

    num_cols = df_eda.select_dtypes(
        include="number"
    ).columns

    cat_cols = df_eda.select_dtypes(
        include=["string", "category", "object"]
    ).columns

    date_cols = df_eda.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns

    # Mostramos las columnas encontradas
    print("VARIABLES NUMERICAS:\n\n", num_cols)
    print('=' * 100)

    print("\nVARIABLES CATEGORICAS:\n\n", cat_cols)
    print('=' * 100)

    print("\nVARIABLES DATETIME(FECHA):\n\n", date_cols)

    # --------------------------------------------------
    # ESTADISTICAS BASICAS
    # --------------------------------------------------

    print("\n========== ESTADÍSTICAS BÁSICAS ==========\n")

    # Variables numericas
    if len(num_cols) > 0:

        print("VARIABLES NUMERICAS:")

        print(
            df_eda[num_cols]
            .describe()
            .T
            .round(2)
        )

    # Variables categoricas
    if len(cat_cols) > 0:

        print("\nVARIABLES CATEGORICAS:")

        print(
            df_eda[cat_cols]
            .describe()
            .T
        )

    # Variables datetime
    if len(date_cols) > 0:

        print("\nVARIABLES DATETIME(FECHA):")

        print(
            df_eda[date_cols]
            .describe()
            .T
        )

    # --------------------------------------------------
    # ANALISIS DE VARIABLES CATEGORICAS
    # --------------------------------------------------

    if len(cat_cols) > 0:

        print(
            "\n========== ANALISIS DE VARIABLES CATEGORICAS ==========\n"
        )

        for col in cat_cols:

            print(
                f"\n----------- ESTAMOS ANALIZANDO: '{col}' ----------\n"
            )

            print("Valores únicos:")

            print(
                df_eda[col].unique()
            )

            print("\nFrecuencia de los valores:")

            print(
                df_eda[col].value_counts()
            )

    # --------------------------------------------------
    # COUNTPLOT
    # --------------------------------------------------

    if len(cat_cols) > 0:

        print(
            "\n============== COUNTPLOT ==============\n"
            "(REPRESENTACIÓN DE UNIVARIABLES CATEGÓRICAS)"
        )

        # Si tiene demasiadas categorias,
        # no hacemos el grafico
        cat_cols_plot = [
            col for col in cat_cols
            if df_eda[col].nunique() <= 200
        ]

        # Mostramos las columnas que no se van a representar
        for col in cat_cols:

            if df_eda[col].nunique() > 200:

                print(
                    f"Columna {col} tiene demasiadas "
                    f"categorias: {df_eda[col].nunique()}"
                )

        # Creamos los graficos solo si hay columnas que representar
        if len(cat_cols_plot) > 0:

            n_graficos = len(cat_cols_plot)
            ncols = 2
            nrows = (n_graficos + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(8 * ncols, 5 * nrows)
            )

            axes = np.atleast_1d(axes).flatten()

            for ax, col in zip(axes, cat_cols_plot):

                sns.countplot(
                    x=df_eda[col],
                    order=df_eda[col].value_counts().index,
                    ax=ax
                )

                ax.set_title(f"Distribución de {col}")
                ax.tick_params(axis="x", rotation=90)

            # Ocultamos ejes sobrantes
            for ax in axes[n_graficos:]:
                ax.set_visible(False)

            plt.tight_layout()
            plt.show()

    # --------------------------------------------------
    # HISTOGRAMAS
    # --------------------------------------------------

    if len(num_cols) > 0:
        print(
            "\n============== HISTOGRAMAS ==============\n"
            "(REPRESENTACION DE UNIVARIABLES NUMERICAS):"
        )

        n_graficos = len(num_cols)
        ncols = 3
        nrows = (n_graficos + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5 * ncols, 3.5 * nrows)
        )

        axes = np.atleast_1d(axes).flatten()

        for ax, col in zip(axes, num_cols):

            sns.histplot(
                df_eda[col],
                bins=20,
                ax=ax
            )

            ax.set_title(col)

        # Ocultamos ejes sobrantes
        for ax in axes[n_graficos:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # BOXPLOTS
    # --------------------------------------------------

    if len(num_cols) > 0:

        print(
            "\n============== BOXPLOTS ==============\n"
            "(REPRESENTACION DE UNIVARIABLES NUMERICAS - OUTLIERS)"
        )

        n_graficos = len(num_cols)
        ncols = 1
        nrows = (n_graficos + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(10 * ncols, 2 * nrows)
        )

        axes = np.atleast_1d(axes).flatten()

        for ax, col in zip(axes, num_cols):

            sns.boxplot(
                x=df_eda[col],
                ax=ax
            )

            ax.set_title(col)

        # Ocultamos ejes sobrantes
        for ax in axes[n_graficos:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()



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


def eda_categoricas(df, cols_excluir=None):
    """
    EDA de las columnas categóricas: estadísticos + valores únicos/frecuencia
    + countplots (se omite el gráfico de las columnas con más de 200 categorías).
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

    for col in cat_cols:
        print(f"\n----------- {col} -----------")
        print('Valores únicos:', df_eda[col].unique())
        print('Frecuencia:')
        print(df_eda[col].value_counts())

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





# ============================================================================
# 2. EDA UNIVARIANTE (Análisis univariante)
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

    print(df[col].value_counts())
    print("."*40)
    print("PORCENTAJES")
    print(df[col].value_counts(normalize=True).mul(100).round(2))

    # --------------------------------------------------
    # HISTPLOT num
    # --------------------------------------------------
def recuento_valores(df, col):
    """
    Frecuencias absolutas y relativas de una variable discreta.
    """
    print("\nVALORES")
    print(df[col].value_counts())

    print("\nPORCENTAJES")
    print(df[col].value_counts(normalize=True).mul(100).round(2))


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

    print(f"\nValor mínimo: {df[col].min():,.2f}")
    print(f"Valor máximo: {df[col].max():,.2f}")
    print(f"Media: {df[col].mean():,.2f}")
    print(f"Mediana: {df[col].median():,.2f}")

    # --------------------------------------------------
    # BOXPLOT + DETECCIÓN DE OUTLIERS num (Outliers)
    # --------------------------------------------------

def boxplot(df, col,n=None):
    """
    Distribución de una variable numérica.
    Dibuja un boxplot y detecta outliers mediante el criterio IQR.

    Muestra:
    - Boxplot
    - Número y porcentaje de outliers
    - Rango considerado normal
    - Frecuencia de los valores outliers
    - Porcentaje de los valores outliers
    """

        # -------------------------
        # BOXPLOT
        # -------------------------
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot de {col}")
    plt.xlabel(col)

    plt.tight_layout()
    plt.show()

        # -------------------------
        # CÁLCULO IQR
        # -------------------------
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr

        # -------------------------
        # DETECTAR OUTLIERS
        # -------------------------
    outliers = df[
        (df[col] < inferior) | 
        (df[col] > superior)
    ][col]

    n_outliers = len(outliers)
    porcentaje = n_outliers / len(df) * 100

        # -------------------------
        # RESUMEN
        # -------------------------
    print(f"{col}: {n_outliers} outliers ({porcentaje:.2f}%)")
    print(
        f"Rango normal: "
        f"[{inferior:.2f}, {superior:.2f}]"
    )

    print("." * 40)

        # -------------------------
        # VALORES DE LOS OUTLIERS
        # -------------------------
    print("VALORES OUTLIERS")
    print(outliers.value_counts().head(n))

    print("." * 40)

        # -------------------------
        # PORCENTAJES
        # -------------------------
    print("PORCENTAJES")
    print(
        outliers
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .head(n)
    )

    return n_outliers

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

    print("Primer registro:")
    print(df[col_fecha].min())

    print("\nÚltimo registro:")
    print(df[col_fecha].max())

    return None


# ============================================================================
# 3. EDA BIVARIANTE — (Analisis Bivariante)
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


def barplot(df, col_cat, col_num, estimator="mean", errorbar=None):
    """
    Estadístico (media, mediana, suma...) de una variable numérica
    según una variable categórica.
    Dibuja un barplot con ese estadístico.

    estimator: "mean", "median", "sum", etc.
    """

    plt.figure(figsize=(10, 5))

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
    # BOXPLOT - BIVARIABLES cat-num
    # --------------------------------------------------

def boxplot_bivar(df, col_cat, col_num):
    """
    Distribución de una variable numérica según una variable categórica.
    Dibuja un boxplot (muestra mediana, dispersión y outliers).
    """
    sns.boxplot(data=df, x=col_cat, y=col_num)
    plt.title(f"Boxplot de {col_num} vs {col_cat}")
    plt.xlabel(col_cat)
    plt.ylabel(col_num)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

    resumen = df.groupby(col_cat)[col_num].describe().round(2)
    print(f"Resumen de {col_num} por {col_cat}:")
    print(resumen)

    return None

    # --------------------------------------------------
    # COUNTPLOT - HUE cat o num binarias-hue(cat) 
    # --------------------------------------------------


def countplot_hue(df, col_cat, hue, normalize=False, sort=True):
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
    plt.figure(figsize=(10, 5))

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
# 4. EDA GLOBAL 
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
# CORRELACIÓN CON OBJETIVO
# --------------------------------------------------
# ============================================================================
# 10. CORRELACIÓN CON LA VARIABLE OBJETIVO (clasificación)
# de sp_utils CORRELACIÓN CON LA VARIABLE OBJETIVO (clasificación)
# ============================================================================
def corr_objetivo(df, target):
    """
    Calcula la correlación de todas las variables numéricas
    respecto a una variable objetivo binaria.

    Parameters
    ----------
    df : pd.DataFrame

    target : str
        Variable objetivo numérica (0/1).

    Returns
    -------
    pd.Series
        Correlaciones ordenadas por valor absoluto.
    """

    corr = (
        df.corr(numeric_only=True)[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
    )

    print(f"Correlación con {target}:")
    print(corr.round(3))

    return None
# Si queremos graficar, cambiar None por corr
