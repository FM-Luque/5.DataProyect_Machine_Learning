"""
sp_abtest.py
================
Caja de herramientas de Fase 4 — Análisis Inferencial.

Tres bloques:

  0) EXPLORACIÓN INICIAL:
     clasificar_columnas() / exploracion_grupos()

  1) FLUJO DE COMPARACIÓN DE GRUPOS (sigue el diagrama ampliado):
     normalidad() -> homocedasticidad_resumen()/
     / ttest_dos_grupos() / mannwhitneyu()
     / anova_tukey() / kruskal() / decidir_test()

  2) ANEXO — tests de relación entre variables (no comparan grupos):
     intervalo_confianza_media() / chi_cuadrado_independencia()
     / correlacion_regresion()

Todas las funciones asumen que le pasas el DataFrame ya al nivel correcto
(df = nivel pedido / order_id, clientes = nivel cliente / id_cliente).
Repasa la Guía de claves antes de decidir qué tabla usar.
"""

# Tratamiento de Datos
import pandas as pd
import numpy as np
from IPython.display import display

# Visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Para que se muestren todas las columnas al inspeccionar los DataFrames
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# ============================================================================
# 0. EXPLORACIÓN INICIAL
# ============================================================================

    # ============================================================================
    # 1. CLASIFICAR COLUMNAS CONTROL - METRICAS 
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
    # 2. EXPLORACIÓN DE GRUPOS 
    # ============================================================================

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
# 1. FLUJO DE COMPARACIÓN DE GRUPOS
# ============================================================================

def normalidad(df, lista_metricas, n_max=5000, random_state=42):
    """Paso 1 del diagrama. Shapiro-Wilk sobre cada métrica.
    Si la columna tiene más de n_max filas, se testea sobre una muestra
    aleatoria de tamaño n_max para evitar el warning de scipy y mantener
    la fiabilidad del test (Shapiro pierde precisión con N > 5000)."""

    for metrica in lista_metricas:

        datos = df[metrica]

        if len(datos) > n_max:
            datos = datos.sample(n_max, random_state=random_state)

        statistic, pvalue = stats.shapiro(datos)

        if pvalue > 0.05:
            print(
                f'Para la columna {metrica.upper()} '
                f'los datos SÍ siguen una distribución normal'
            )
        else:
            print(
                f'Para la columna {metrica.upper()} '
                f'los datos NO siguen una distribución normal'
            )


def homocedasticidad_resumen(df, col_control, metricas, alpha=0.05):
    """
    Evalúa la homocedasticidad mediante el test de Levene y muestra
    únicamente las métricas que presentan varianzas homogéneas entre
    los grupos definidos por la columna de control.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame a analizar.

    col_control : str
        Columna categórica que define los grupos.

    metricas : str o list
        Métrica o lista de métricas numéricas.

    alpha : float, default=0.05
        Nivel de significación.

    Returns
    -------
    list
        Lista de métricas que cumplen homocedasticidad.
    """

    if isinstance(metricas, str):
        metricas = [metricas]

    metricas_homocedasticas = []

    for metrica in metricas:

        muestras = [
            grupo[metrica].dropna()
            for _, grupo in df.groupby(col_control)
        ]

        try:

            _, p_value = stats.levene(*muestras)

            if p_value > alpha:
                metricas_homocedasticas.append(metrica)

        except Exception:
            continue

    if metricas_homocedasticas:

        print(
            f"Para la columna {col_control.upper()} las varianzas "
            f"de las siguientes métricas SÍ son homogéneas. "
            
        )
        print("SÍ hay HOMOCEDASTICIDAD\n")
        print(metricas_homocedasticas)
        print("\n")
    else:

        print(
            f"Para la columna {col_control.upper()} NO se han "
            f"encontrado métricas con HOMOCEDASTICIDAD."
        )
        print("\n")
    return metricas_homocedasticas

def ttest_dos_grupos(df, col_control, lista_metricas):
    """Rama: 2 grupos, normal, homocedástico.
    Sustituye a 'z-score' del diagrama — con datos de muestra (no con sigma
    poblacional conocida) el test correcto es t-test.
    equal_var=False (Welch) por defecto: no exige varianzas exactamente iguales.
    p < 0.05  → hay evidencia de diferencias significativas
    p ≥ 0.05  → no hay evidencia de diferencias significativas
    """

    for metrica in lista_metricas:

        valores_control = df[col_control].unique()

        grupo_a = df[df[col_control] == valores_control[0]][metrica]
        grupo_b = df[df[col_control] == valores_control[1]][metrica]

        t_stat, pvalue = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)

        print(f't={t_stat:.4f}, p={pvalue:.4f}')

        if pvalue > 0.05:
            print(f'Para la métrica {metrica.upper()}, las medias SI son iguales, es decir, NO hay diferencias significativas entre grupos')
        else:
            print(f'Para la métrica {metrica.upper()}, las medias NO son iguales, es decir, SI hay diferencias significativas entre grupos')


def mannwhitneyu(df, col_control, lista_metricas):
    """Rama: 2 grupos, no normal y/o no homocedástico. Compara medianas.
        p < 0.05  → hay evidencia de diferencias significativas
        p ≥ 0.05  → no hay evidencia de diferencias significativas
    """

    for metrica in lista_metricas:

        valores_control = df[col_control].unique()

        control = df[df[col_control] == valores_control[0]][metrica]
        test = df[df[col_control] == valores_control[1]][metrica]

        statistic, pvalue = stats.mannwhitneyu(control, test)

        print(f'U={statistic:.4f}, p={pvalue:.4f}')

        if pvalue > 0.05:
            print(f'Para la métrica {metrica.upper()}, las medianas SI son iguales, es decir, NO hay deferencias significativas entre grupos')
        else:
            print(f'Para la métrica {metrica.upper()}, las medianas NO son iguales, es decir, SI hay deferencias significativas entre grupos')


def anova_tukey(df, col_control, lista_metricas, alpha=0.05):
    """Rama: 3+ grupos, normal, homocedástico. ANOVA + post-hoc Tukey
    si el ANOVA global sale significativo (dice qué pares difieren).
    
    p < 0.05  → hay evidencia de diferencias significativas
    p ≥ 0.05  → no hay evidencia de diferencias significativas
    """

    for metrica in lista_metricas:

        grupos = [g[metrica].values for _, g in df.groupby(col_control)]

        f_stat, pvalue = stats.f_oneway(*grupos)

        print(f'F={f_stat:.4f}, p={pvalue:.4f}')

        if pvalue > 0.05:
            print(f'Para la métrica {metrica.upper()}, las medias SI son iguales entre los grupos de {col_control} (ANOVA no significativo)')
        else:
            print(f'Para la métrica {metrica.upper()}, al menos un grupo de {col_control} tiene una media distinta (ANOVA significativo)')
            print('Post-hoc Tukey (qué pares difieren):')
            tukey = pairwise_tukeyhsd(df[metrica], df[col_control], alpha=alpha)
            print(tukey.summary())


def kruskal(df, col_control, lista_metricas):
    """Rama: 3+ grupos, no normal y/o no homocedástico.
    Equivalente no paramétrico de ANOVA — cierra la 4ª rama del flujo,
    ya que mannwhitneyu() solo sirve para 2 grupos.
    Post-hoc equivalente a Tukey (si hace falta): test de Dunn,
    disponible en la librería scikit-posthocs (no incluida aquí).
        p < 0.05  → hay evidencia de diferencias significativas
        p ≥ 0.05  → no hay evidencia de diferencias significativas
    """

    for metrica in lista_metricas:

        grupos = [g[metrica].values for _, g in df.groupby(col_control)]

        statistic, pvalue = stats.kruskal(*grupos)

        print(f'H={statistic:.4f}, p={pvalue:.4f}')

        if pvalue > 0.05:
            print(f'Para la métrica {metrica.upper()}, las medianas SI son iguales entre los grupos de {col_control} (Kruskal-Wallis no significativo)')
        else:
            print(f'Para la métrica {metrica.upper()}, al menos un grupo de {col_control} tiene una mediana distinta (Kruskal-Wallis significativo)')


def decidir_test(df, col_control, lista_metricas, n_max=5000, random_state=42, alpha=0.05):
    """Orquestador: recorre el diagrama completo (normalidad -> homocedasticidad
    -> nº de grupos) y llama automáticamente al test que corresponda para
    cada métrica. Útil para no tener que decidir a mano cada vez."""

    n_grupos = df[col_control].nunique()

    for metrica in lista_metricas:

        datos_normalidad = df[metrica]
        if len(datos_normalidad) > n_max:
            datos_normalidad = datos_normalidad.sample(n_max, random_state=random_state)

        es_normal = stats.shapiro(datos_normalidad)[1] > 0.05

        grupos_vals = [g[metrica].values for _, g in df.groupby(col_control)]
        es_homocedastico = stats.levene(*grupos_vals)[1] > 0.05

        print(f'--- {metrica.upper()} | normalidad={es_normal} | homocedastico={es_homocedastico} | n_grupos={n_grupos} ---')

        if es_normal and es_homocedastico and n_grupos == 2:
            ttest_dos_grupos(df, col_control, [metrica])
        elif es_normal and es_homocedastico and n_grupos > 2:
            anova_tukey(df, col_control, [metrica], alpha=alpha)
        elif n_grupos == 2:
            mannwhitneyu(df, col_control, [metrica])
        else:
            kruskal(df, col_control, [metrica])

        print('=' * 100)


# ============================================================================
# 2. ANEXO — TESTS DE RELACIÓN ENTRE VARIABLES (no comparan grupos)
# ============================================================================

def intervalo_confianza_media(df, lista_metricas, confianza=0.95):
    """IC para la media de cada métrica (por defecto al 95%)."""

    for metrica in lista_metricas:
        media = df[metrica].mean()
        sem = stats.sem(df[metrica])
        ic = stats.t.interval(confianza, len(df[metrica]) - 1, loc=media, scale=sem)
        print(f'{metrica.upper()} -> media={media:.2f}  IC {int(confianza*100)}%=({ic[0]:.2f}, {ic[1]:.2f})')


def chi_cuadrado_independencia(df, col_a, col_b):
    """Chi-cuadrado de independencia entre dos variables categóricas."""

    tabla = pd.crosstab(df[col_a], df[col_b])
    chi2, pvalue, dof, expected = stats.chi2_contingency(tabla)

    print(f'chi2={chi2:.3f}, p={pvalue:.4f}, dof={dof}')
    if pvalue < 0.05:
        print(f'SI hay relación entre {col_a} y {col_b}')
    else:
        print(f'NO hay evidencia de relación entre {col_a} y {col_b}')


def correlacion_regresion(df, col_x, col_y):
    """Correlación (Pearson y Spearman) + regresión lineal simple col_y ~ col_x.
    Recuerda: si una variable es a nivel cliente, agrégala primero por id_cliente."""

    r_pearson, p_pearson = stats.pearsonr(df[col_x], df[col_y])
    r_spearman, p_spearman = stats.spearmanr(df[col_x], df[col_y])
    print(f'Pearson r={r_pearson:.3f} (p={p_pearson:.4f})')
    print(f'Spearman rho={r_spearman:.3f} (p={p_spearman:.4f})')

    modelo = smf.ols(f'{col_y} ~ {col_x}', data=df).fit()
    print(modelo.summary())
    return modelo




