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

    # Aceptar una métrica o varias
    if isinstance(lista_metricas, str):
        lista_metricas = [lista_metricas]

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
    # Aceptar una métrica o varias
    if isinstance(lista_metricas, str):
        lista_metricas = [lista_metricas]

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



def anova_tukey(df, lista_controles, lista_metricas, alpha=0.05):
    """Rama: 3+ grupos, normal, homocedástico. ANOVA + post-hoc Tukey
    si el ANOVA global sale significativo (dice qué pares difieren).
    
    p < 0.05  → hay evidencia de diferencias significativas
    p ≥ 0.05  → no hay evidencia de diferencias significativas
    """
    if isinstance(lista_controles, str):
        lista_controles = [lista_controles]

    if isinstance(lista_metricas, str):
        lista_metricas = [lista_metricas]

    resultados = []

    for control in lista_controles:

        for metrica in lista_metricas:

            grupos = [
                g[metrica].values
                for _, g in df.groupby(control)
            ]

            f_stat, pvalue = stats.f_oneway(*grupos)

            resultados.append({
                'control': control,
                'metrica': metrica,
                'F': round(f_stat, 4),
                'pvalue': round(pvalue, 4),
                'significativo': pvalue < alpha
            })

            if pvalue < alpha:

                print(f'\n{"="*80}')
                print(f'{control.upper()} | {metrica.upper()}')
                print(f'{"="*80}')

                tukey = pairwise_tukeyhsd(
                    df[metrica],
                    df[control],
                    alpha=alpha
                )

                print(tukey.summary())

    return pd.DataFrame(resultados)

def kruskal(df, lista_controles, lista_metricas):
    """Rama: 3+ grupos, no normal y/o no homocedástico.
    Equivalente no paramétrico de ANOVA — cierra la 4ª rama del flujo,
    ya que mannwhitneyu() solo sirve para 2 grupos.
    Post-hoc equivalente a Tukey (si hace falta): test de Dunn,
    disponible en la librería scikit-posthocs (no incluida aquí).
        p < 0.05  → hay evidencia de diferencias significativas
        p ≥ 0.05  → no hay evidencia de diferencias significativas
    """

    if isinstance(lista_controles, str):
        lista_controles = [lista_controles]

    if isinstance(lista_metricas, str):
        lista_metricas = [lista_metricas]

    resultados = []

    for control in lista_controles:

        for metrica in lista_metricas:

            grupos = [
                g[metrica].values
                for _, g in df.groupby(control)
            ]

            h, pvalue = stats.kruskal(*grupos)

            resultados.append({
                'control': control,
                'metrica': metrica,
                'H': round(h, 4),
                'pvalue': round(pvalue, 4),
                'significativo': pvalue < 0.05
            })

    return pd.DataFrame(resultados)



def decidir_test(
    df,
    lista_controles,
    lista_metricas,
    n_max=5000,
    random_state=42,
    alpha=0.05
):
    """
    Decide automáticamente qué test aplicar
    según normalidad, homocedasticidad y nº de grupos.

    Devuelve DataFrame resumen.
    """

    import pandas as pd
    from scipy import stats

    if isinstance(lista_controles, str):
        lista_controles = [lista_controles]

    if isinstance(lista_metricas, str):
        lista_metricas = [lista_metricas]

    resultados = []

    for control in lista_controles:

        n_grupos = df[control].nunique()

        for metrica in lista_metricas:

            # -----------------------
            # Normalidad
            # -----------------------

            datos_normalidad = df[metrica]

            if len(datos_normalidad) > n_max:
                datos_normalidad = datos_normalidad.sample(
                    n=n_max,
                    random_state=random_state
                )

            p_normalidad = stats.shapiro(
                datos_normalidad
            )[1]

            es_normal = p_normalidad > alpha

            # -----------------------
            # Homocedasticidad
            # -----------------------

            grupos_vals = [
                g[metrica].values
                for _, g in df.groupby(control)
            ]

            p_levene = stats.levene(
                *grupos_vals
            )[1]

            es_homocedastico = p_levene > alpha

            # -----------------------
            # Decisión
            # -----------------------

            if es_normal and es_homocedastico and n_grupos == 2:

                stat, pvalue = stats.ttest_ind(
                    *grupos_vals,
                    equal_var=False
                )

                test = "ttest"

            elif es_normal and es_homocedastico and n_grupos > 2:

                stat, pvalue = stats.f_oneway(
                    *grupos_vals
                )

                test = "anova"

            elif n_grupos == 2:

                stat, pvalue = stats.mannwhitneyu(
                    *grupos_vals,
                    alternative="two-sided"
                )

                test = "mannwhitney"

            else:

                stat, pvalue = stats.kruskal(
                    *grupos_vals
                )

                test = "kruskal"

            resultados.append({
                'control': control,
                'metrica': metrica,
                'test': test,
                'estadistico': round(stat, 4),
                'pvalue': round(pvalue, 4),
                'normal': es_normal,
                'homocedastico': es_homocedastico,
                'n_grupos': n_grupos,
                'significativo': pvalue < alpha
            })

    return pd.DataFrame(resultados)





