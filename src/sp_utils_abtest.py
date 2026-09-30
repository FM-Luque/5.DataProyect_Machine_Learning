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
    