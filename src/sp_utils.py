"""
sp_utils.py
================
Cuaderno de referencia (bitácora) — NO es un módulo con funciones para
importar, como el resto de sp_*.py. Es una chuleta de comandos sueltos
de pandas que se usan constantemente en EDA, A/B testing y limpieza,
para no tener que buscarlos cada vez.

Cómo usarlo: abre este archivo, busca la sección que necesites
(Ctrl+F por el título), copia la línea que te sirva y pégala en tu
notebook, cambiando "df" por el nombre real de tu DataFrame.

Todas las líneas de ejemplo están comentadas con "#" a propósito,
para que este archivo no dé error si alguna vez lo ejecutas por
accidente (no hay ningún "df" real definido aquí).

Índice
------
 0. Exploración inicial
 1. Tipos de dato
 2. Valores y conteos
 3. Nulos
 4. Recorrer columnas
 5. Seleccionar filas/columnas (iloc / loc)
 6. Filtrar filas por condición
    [Nota] Orden de trabajo para EDA por subgrupo (correlación global +
    groupby/crosstab por columna)
 7. Agrupaciones categórica + numérica (groupby)
 8. Agrupaciones categóricas (crosstab)
 9. Relación entre variables numéricas (correlación general)
10. Correlación con la variable objetivo (clasificación)
11. Ordenar
12. Valor centinela (separar un código especial de un dato real)
13. Reiniciar / cambiar el índice
14. Filtrado por lista o por rango
15. Texto con .str
16. Regex básico para limpieza
17. Aplicar una función propia (.apply())
18. Groupby con varios estadísticos a la vez
19. Unir DataFrames (concat() y merge())
20. Tabla dinámica (pivot_table())
21. Duplicados
22. Valores extremos rápidos (nlargest / nsmallest)
23. Reglas de negocio / coherencia entre columnas (Fase 2, paso 6.5)
"""

# df = sc.leer_csv(r"archivo.csv",parse_date= [' ' ,' '])


# ============================================================================
# 0. EXPLORACIÓN INICIAL — primer vistazo al DataFrame
# ============================================================================

# df.head()
#   Muestra las primeras 5 filas. df.head(10) -> las primeras 10.

# df.tail()
#   Muestra las últimas 5 filas. Útil para ver si el final del archivo
#   tiene basura (filas de totales, líneas vacías, etc.).

# df.sample(5)
#   Muestra 5 filas AL AZAR, no las primeras. Mejor que head() para
#   hacerte una idea real del contenido, porque head() solo te enseña
#   siempre el principio (que a veces no es representativo de todo).

# df.shape
#   Devuelve (nº_filas, nº_columnas). Ej: (43000, 27).

# df.info()
#   Resumen rápido: nombre de cada columna, cuántos valores NO nulos
#   tiene, y su tipo de dato. El primer comando que se suele lanzar
#   nada más cargar un dataset.

# df.describe()
#   Estadísticos (media, mínimo, máximo, percentiles...) de las
#   columnas NUMÉRICAS. Para las de texto usa df.describe(include='str').

# df.dtypes
#   Lista el tipo de dato de cada columna, sin más estadísticos.
#   Como el "tipo" de tu propio reporte_calidad(), pero sin ejemplos.

# df.columns.tolist()
# Devuelve las columnas del Dataframe, en forma de listado

# df['col'].mode()
#   Devuelve el valor (o valores) más frecuentes de la columna.
#   Equivale a la moda en estadística.

# df['col'].mode()[0]
#   Devuelve la moda como valor escalar.
#   Muy útil para imputar variables categóricas.


# ============================================================================
# 1. TIPOS DE DATO — seleccionar columnas según su tipo
# ============================================================================

# df.select_dtypes(include='number')
#   Te devuelve un DataFrame solo con las columnas numéricas
#   (int, float). Muy usado antes de un .corr() o un groupby().mean().

# df.select_dtypes(include='string')
#   Solo las columnas de texto. (En pandas < 3.0 sería include='object').

# df.select_dtypes(include='number').columns
#   Si solo quieres la LISTA de nombres de esas columnas, no los datos.
#   Ej: columnas_numericas = df.select_dtypes(include='number').columns


# ============================================================================
# 2. VALORES Y CONTEOS — qué hay dentro de una columna
# ============================================================================

# df['col'].unique()
#   Lista de valores DISTINTOS que aparecen en la columna, sin contar
#   cuántas veces sale cada uno.

# df['col'].nunique()
#   Cuántos valores distintos hay (un número, no la lista).

# df['col'].value_counts()
#   Cada valor distinto, con cuántas veces aparece. Ordenado de más a
#   menos frecuente. El comando más usado para ver categorías raras
#   o mal escritas (ej. 'Madrid' y 'MADRID' como si fueran distintas).

# df['col'].value_counts(dropna=False)
#   Igual, pero contando también los nulos como una categoría más.

# df['col'].value_counts(normalize=True)
#   Igual, pero en % en vez de en número de filas. Útil para ver el
#   balance de la variable objetivo (ej. % de 'yes' vs % de 'no').


# ============================================================================
# 3. NULOS — detectar huecos
# ============================================================================

# df.isna()
#   Devuelve el mismo DataFrame pero con True/False según si cada
#   celda es nula o no. Rara vez se usa así solo, casi siempre
#   encadenado con .sum() o .any().

# df.isna().sum()
#   Nº de nulos por columna. Es la base de tu propia buscar_nulos().

# df.isna().sum()[df.isna().sum() > 0]
#   Nº de nulos por columna. mayor que 0

# df['col'].isna().sum()
#   Nº de nulos, pero de UNA sola columna.

# df.isna().any()
#   Por columna, True si tiene AL MENOS un nulo, False si no tiene
#   ninguno. Rápido para saber "¿qué columnas tocar?" de un vistazo.

# df.notna()
#   Lo contrario de isna(): True donde SÍ hay dato.

# df[['educacion', 'ocupacion', 'estado_civil']].mode()

#	Eliminar filas con valores nulos en columnas específicas:
# df.dropna(subset=['columna'], inplace=True)

#	Eliminar columnas con una alta proporción de valores nulos (por ejemplo, más del 50%):
# df.dropna(thresh=len(df)*0.5, axis=1, inplace=True)


# ============================================================================
# 4. RECORRER COLUMNAS — bucles típicos
# ============================================================================

# for col in df.columns:
#     print(col)
#   Recorre el NOMBRE de cada columna, una a una. Se usa mucho
#   combinado con un if para actuar solo sobre ciertas columnas.

# for col in df.columns:
#     print(col, df[col].dtype)
#   Igual, pero mostrando también el tipo de cada una.

# for col in df.select_dtypes(include='number').columns:
#     print(col)
#   Igual, pero recorriendo SOLO las columnas numéricas (combinando
#   los puntos 1 y 4 de esta chuleta).


# ============================================================================
# 5. SELECCIONAR FILAS/COLUMNAS — iloc y loc
# ============================================================================

# df.iloc[0]
#   La fila en la posición 0 (la primera), sea cual sea su índice real.
#   "iloc" = por POSICIÓN (un número).

# df.iloc[0:5]
#   Las 5 primeras filas (posiciones 0 a 4). Como head(5), pero con
#   sintaxis de slicing.

# df.iloc[:, 0:3]
#   Todas las filas, pero solo las 3 primeras columnas (por posición).

# df.loc[3]
#   La fila cuyo ÍNDICE es 3 (no la 4ª posición necesariamente, si el
#   índice se ha reordenado o tiene huecos). "loc" = por ETIQUETA.

# df.loc[df['edad'] > 50]
#   Filas donde la columna 'edad' es mayor que 50. Esta es la forma
#   más habitual de FILTRAR filas por una condición.

# df.loc[df['edad'] > 50, ['edad', 'ocupacion']]
#   Igual, pero quedándote solo con esas dos columnas del resultado.


# ============================================================================
# 6. FILTRAR FILAS POR CONDICIÓN
# ============================================================================

# df[df['objetivo'] == 'yes']
#   Solo las filas donde esa columna vale exactamente 'yes'.

# df[(df['edad'] > 30) & (df['objetivo'] == 'yes')]
#   Varias condiciones a la vez: usa "&" (y), "|" (o) — NO uses
#   "and"/"or" de Python normal, con DataFrames no funcionan igual.
#   Cada condición entre paréntesis es obligatorio.

# df[df['ocupacion'].isin(['housemaid', 'services'])]
#   Filas donde la columna tiene UNO de varios valores de una lista.


# ============================================================================
# [NOTA] ORDEN DE TRABAJO PARA EDA POR SUBGRUPO
# ============================================================================
# 1. df_origen.corr(numeric_only=True) -> visión global (ya hecho)
# 2. por cada subgrupo:
#      - columnas numéricas   -> groupby('objetivo').mean() / describe()
#      - columnas categóricas -> pd.crosstab(col, objetivo, normalize='index')


# ============================================================================
# 7. AGRUPACIONES CATEGÓRICA + NUMÉRICA — groupby (clave en EDA y A/B testing)
# ============================================================================

# df.groupby('ocupacion')['ingresos'].mean()
#   Media de 'ingresos' para cada valor distinto de 'ocupacion'.
#   Es el comando más usado para comparar un número entre grupos.

# df.groupby('ocupacion')['ingresos'].agg(['mean', 'median', 'std', 'count'])
#   Igual, pero varios estadísticos a la vez de golpe.

# df.groupby('objetivo')['edad'].describe()
#   El describe() completo, pero separado por grupo. Muy típico en
#   A/B testing: comparar la variable numérica entre 'yes' y 'no'.

# df.groupby('hoja_origen')['id'].nunique()
#   Nº de clientes ÚNICOS por año (lo que vimos hace unos mensajes,
#   recuerda: nunique(), nunca sum(), si la columna es texto/id).


# ============================================================================
# 8. AGRUPACIONES CATEGÓRICAS — crosstab
# ============================================================================

# pd.crosstab(df['ocupacion'], df['objetivo'])
#   Tabla cruzada: cuenta cuántas filas hay de cada combinación entre
#   dos columnas categóricas. Muy usado para ver si una categoría
#   está relacionada con el objetivo (ej. ¿qué ocupación contrata más?).

# pd.crosstab(df['ocupacion'], df['objetivo'], normalize='index')
#   Igual, pero en % por fila, para comparar proporciones en vez de
#   números absolutos (más justo si los grupos tienen tamaños distintos).


# ============================================================================
# 9. RELACIÓN ENTRE VARIABLES NUMÉRICAS — EDA y A/B testing
# ============================================================================

# df.corr(numeric_only=True)
#   Matriz de correlación entre TODAS las columnas numéricas.
#   Valores cercanos a 1 o -1 = relación fuerte; cercanos a 0 = poca relación.

# df['col1'].corr(df['col2'])
#   Correlación entre dos columnas concretas, un solo número.


# ============================================================================
# 10. CORRELACIÓN CON LA VARIABLE OBJETIVO (clasificación)
# ============================================================================
#
# df_e.corr(numeric_only=True)['objetivo'].sort_values(ascending=False)
#  
# Si la variable objetivo está codificada como 0/1:
#
# corr_obj = (
#     df.corr(numeric_only=True)['objetivo']
#       .drop('objetivo')
#       .sort_values(key=abs, ascending=Fals)
# )
#
# corr_obj
#
# Devuelve la correlación de todas las variables numéricas respecto a
# la variable objetivo, ordenadas de menor a mayor relación.
#
# Muy útil para detectar variables potencialmente predictivas antes de
# entrenar un modelo.
#
# Si la variable objetivo es texto ('si'/'no'):
#
# df['objetivo_num'] = df['objetivo'].map({'no':0,'si':1})
#
# corr_obj = (
#     df.corr(numeric_only=True)['objetivo_num']
#       .drop('objetivo_num')
#       .sort_values()
# )
# corr_obj
#
# Para representarlo con tu framework:
#
# se.barplot_serie(
#     corr_obj,
#     'Correlación con objetivo',
#     kind='barh'
# )
#
# Recuerda:
# - Pearson mide relación lineal.
# - Correlación ≠ causalidad.
# - Solo funciona correctamente con variables numéricas.


# corr_top10 = corr_objetivo.sort_values(key=abs, ascending=False).head(10).sort_values()

# se.barplot_serie(corr_top10,titulo="Top 10 variables correlacionadas con el objetivo",
#    kind="barh",
#    xlabel="Correlación",
#    ylabel="Variable")


# ============================================================================
# 11. ORDENAR
# ============================================================================

# df.sort_values('ingresos')
#   Ordena el DataFrame por esa columna, de menor a mayor.

# df.sort_values('ingresos', ascending=False)
#   Igual, pero de mayor a menor.

# df.sort_values(['ocupacion', 'ingresos'])
#   Ordena por varias columnas a la vez: primero por 'ocupacion',
#   y dentro de cada ocupación, por 'ingresos'.


# ============================================================================
# 12. VALOR CENTINELA — separar un código especial (ej. 999) de un
#     número real dentro de la misma columna
# ============================================================================

# valor_centinela = 999
#   El código especial que dice "esto no aplica" en tu columna.
# df['contacto_previo'] = np.where(df['col'] == valor_centinela, 'no_contactado', 'contactado')

#   Una única columna categórica: 'no_contactado' si el valor era el
#   código especial, 'contactado' para CUALQUIER otro valor (0, 1, 2,
#   27... da igual el número exacto de días). A diferencia de separar
#   en dos columnas (booleana + número real), aquí se pierde a
#   propósito el detalle de "cuántos días" — útil cuando lo que
#   importa es solo el "hubo contacto antes, sí o no", no cuánto hace.


# ============================================================================
# ============================  V2  =========================================
# Snippets seleccionados de tu resumen "My_Resumen_Pandas_for_Data".
# No están todos los de tu documento a propósito — dejé fuera los que ya
# haces de otra forma en tu framework (join(), assign(), insert(), eval(),
# melt()) por no duplicar, o los que son poco probables que uses ahora
# mismo. Si echas en falta alguno en concreto, dímelo y lo añado.
# ============================================================================


# ============================================================================
# 13. REINICIAR / CAMBIAR EL ÍNDICE
# ============================================================================

# df.reset_index()
#   Vuelve a numerar las filas desde 0, y guarda el índice antiguo como
#   una columna nueva. Se usa mucho DESPUÉS de un groupby(), porque
#   groupby() deja la columna agrupada como índice en vez de columna
#   normal.

# df.reset_index(drop=True)
#   Igual, pero sin guardar el índice antiguo (lo descarta). Es la
#   versión que más se usa: "renumera y olvida el índice viejo".

# df.groupby('ocupacion')['ingresos'].mean().reset_index()
#   El combo típico: agrupas, y al final haces reset_index() para que
#   el resultado vuelva a ser un DataFrame normal con 'ocupacion' como
#   columna (no como índice), fácil de seguir usando o guardar en csv.

# df.set_index('id_cliente')
#   Al revés que reset_index(): convierte una columna EN el índice.
#   Útil si vas a buscar filas muy a menudo por ese valor.


# ============================================================================
# 14. FILTRADO POR LISTA O POR RANGO
# ============================================================================

# df[df['ocupacion'].isin(['housemaid', 'services'])]
#   Filas donde la columna tiene UNO de varios valores de una lista.
#   Más cómodo que encadenar varios "|" (or) uno por uno.

# df[~df['ocupacion'].isin(['housemaid', 'services'])]
#   Lo contrario: filas que NO están en esa lista. El "~" delante
#   invierte la condición (como decir "NO cumple esto").

# df[df['edad'].between(30, 50)]
#   Filas donde 'edad' está entre 30 y 50, ambos incluidos. Más
#   claro que escribir (df['edad'] >= 30) & (df['edad'] <= 50).

# df[df['edad'].between(30, 50, inclusive='left')]
#   Igual, pero sin incluir el 50 (inclusive='right' sería sin el 30,
#   'neither' sin ninguno de los dos extremos).


# ============================================================================
# 15. TEXTO CON .str — lo más usado, aparte de lo que ya tienes en
#     limpiar_texto()/eliminar_acentos()
# ============================================================================

# df['col'].str.len()
#   Longitud (nº de caracteres) de cada valor de texto. Útil para
#   detectar valores raros (ej. códigos postales con longitud rara).

# df['col'].str.split('-')
#   Divide el texto en una lista, cortando por el separador indicado.
#   Ej: '2024-08-15'.split('-') -> ['2024', '08', '15'].

# df['col'].str.contains('texto')
#   True/False según si el texto aparece dentro de la celda. Se usa
#   sobre todo para FILTRAR filas: df[df['col'].str.contains('gmail')]

# df['col'].str.startswith('A')
#   True/False según si el texto empieza por eso. str.endswith('z')
#   es la misma idea, pero mirando el final.


# ============================================================================
# 16. REGEX BÁSICO PARA LIMPIEZA (chuleta rápida, no hace falta saber
#     regex a fondo para usar estos 4 patrones)
# ============================================================================

# df['col'].str.contains(r'^A', na=False)
#   Empieza por "A". El "^" significa "inicio del texto".

# df['col'].str.contains(r'0$', na=False)
#   Termina en "0". El "$" significa "final del texto".

# df['col'].str.replace(r'[^0-9]', '', regex=True)
#   Se queda SOLO con los números, borra cualquier otra cosa
#   (letras, espacios, símbolos). Útil para limpiar teléfonos, códigos.

# df['col'].str.replace(r'\s+', ' ', regex=True)
#   Colapsa varios espacios seguidos en uno solo. Es justo lo que hace
#   por dentro tu propia limpiar_texto().

# Nota: na=False en str.contains() es importante si tu columna tiene
# nulos — sin eso, las filas con NaN dan error o resultado raro en
# vez de simplemente "no cumple la condición".


# ============================================================================
# 17. APLICAR UNA FUNCIÓN PROPIA — .apply()
# ============================================================================

# df['col_nueva'] = df['col'].apply(lambda x: x * 2)
#   Aplica una operación a CADA valor de la columna, uno por uno.
#   Solo lo necesitas cuando la operación no tiene un método de pandas
#   ya hecho (si existe, como .str.lower(), úsalo siempre a él primero,
#   es más rápido que apply()).

# def clasificar_edad(edad):
#     if edad < 30:
#         return 'joven'
#     elif edad < 60:
#         return 'adulto'
#     else:
#         return 'mayor'
#
# df['grupo_edad'] = df['edad'].apply(clasificar_edad)
#   Ejemplo típico: crear una columna categórica nueva a partir de
#   reglas que no existen como método de pandas.


# ============================================================================
# 18. GROUPBY CON VARIOS ESTADÍSTICOS A LA VEZ
# ============================================================================

# df.groupby('ocupacion')['ingresos'].agg(['mean', 'median', 'std', 'count'])
#   Ya la tenías en la sección 7, la repito aquí junto a esta variante:

# df.groupby('ocupacion').agg(
#     ingresos_medio=('ingresos', 'mean'),
#     edad_media=('edad', 'mean'),
# ).reset_index()
#   Agregación con NOMBRE personalizado para cada columna resultado,
#   y sobre columnas DISTINTAS a la vez (aquí, ingresos y edad juntas
#   en la misma tabla). Más claro que el anterior cuando agrupas varias
#   columnas distintas de golpe.


# ============================================================================
# 19. UNIR DATAFRAMES — concat() y merge()
#     (la teoría ya la conoces de tu propio integracion.py, esto es
#     solo la sintaxis mínima para recordarla rápido)
# ============================================================================

# pd.concat([df1, df2], axis=0, ignore_index=True)
#   Apila DataFrames uno debajo del otro (mismas columnas). axis=0 es
#   por defecto. ignore_index=True renumera las filas desde 0, en vez
#   de arrastrar los índices originales de cada trozo.

# df_izq.merge(df_der, on='id_cliente', how='left')
#   Une dos DataFrames por una columna común. how='left' conserva TODAS
#   las filas del de la izquierda, aunque no tengan pareja en el
#   derecho (quedarían con NaN en esas columnas).

# df_izq.merge(df_der, on='id_cliente', how='inner')
#   Solo conserva las filas donde el id_cliente existe en AMBOS
#   DataFrames — las que no coinciden se pierden.


# ============================================================================
# 20. TABLA DINÁMICA — pivot_table()
# ============================================================================

# df.pivot_table(values='ingresos', index='ocupacion', columns='objetivo', aggfunc='mean')
#   El equivalente pandas a una tabla dinámica de Excel: filas =
#   'ocupacion', columnas = 'objetivo', y en cada celda la media de
#   'ingresos' para esa combinación. Muy visual para comparar dos
#   categorías a la vez contra un número.


# ============================================================================
# 21. DUPLICADOS
# ============================================================================

# df.duplicated()
#   True/False por fila: True si esa fila es idéntica a otra anterior.
#   Es la base de tu propia buscar_duplicados()/eliminar_duplicados().

# df.duplicated().sum()
#   Cuántas filas están duplicadas en total (un solo número).

# df[df.duplicated(keep=False)]
#   Muestra TODAS las filas implicadas en un duplicado (tanto la
#   original como la copia), no solo la copia sobrante — útil para
#   comparar ambas antes de decidir cuál conservar.

# df.duplicated(subset=['id_cliente'])
#   Duplicados mirando solo esas columnas, no la fila entera. Útil
#   para detectar un mismo cliente repetido aunque el resto de sus
#   datos varíe.


# ============================================================================
# 22. VALORES EXTREMOS RÁPIDOS — nlargest / nsmallest
# ============================================================================

# df['ingresos'].nlargest(5)
#   Los 5 valores más ALTOS de la columna, ya ordenados. Más directo
#   que sort_values(ascending=False).head(5).

# df['ingresos'].nsmallest(5)
#   Los 5 valores más BAJOS. Igual que nlargest() pero al revés.

# df.nlargest(5, 'ingresos')
#   Igual, pero sobre el DataFrame completo: te da las 5 FILAS enteras
#   con mayor 'ingresos', no solo el valor de esa columna suelto.

# df.nlargest(5, 'ingresos')[['id_cliente', 'ingresos', 'ocupacion']]
#   Lo mismo, quedándote solo con las columnas que te interesa ver.


# ============================================================================
# 23. REGLAS DE NEGOCIO / COHERENCIA ENTRE COLUMNAS (Fase 2, paso 6.5)
#     Detecta filas donde varias columnas, cada una válida por
#     separado, no tienen sentido JUNTAS. Las reglas cambian con cada
#     dataset — esto es una plantilla a adaptar, no una función fija.
# ============================================================================

# reglas = {
#     'nombre_de_la_regla_1': mascara_booleana_1,
#     'nombre_de_la_regla_2': mascara_booleana_2,
# }
#   Cada máscara es una condición booleana entre columnas, del tipo
#   df['col_a'] > df['col_b'], o mezclando categórica+numérica como
#   (df['contacto_previo'] == 'no_contactado') & (df['dias_contacto'] != 999).
#   Estas comprobaciones son VECTORIZADAS: se calculan sobre TODO el
#   DataFrame de golpe, sin importar si tiene 100 o 100.000 filas.

# inconsistencias = []
# resumen_reglas = {}

# for nombre_regla, mask in reglas2.items():
   # indices = df_sl.index[mask]
   # inconsistencias.extend([(i, nombre_regla) for i in indices])
   # resumen_reglas[nombre_regla] = mask.sum()

# inconsistencias_df = pd.DataFrame(inconsistencias,columns=['indice', 'regla'])

# pd.Series(resumen_reglas).sort_values(ascending=False)

#   Junta todas las filas que rompen alguna regla, con qué regla
#   rompieron cada una.

# inconsistencias_df['regla'].value_counts()
#   Cuántas inconsistencias hay de cada tipo — así decides el
#   tratamiento POR REGLA (no fila a fila): ¿se descartan, se marcan,
#   se corrigen? No es necesario ni recomendable revisar cada fila
#   una a una, solo una muestra pequeña para entender el patrón:
#   df.loc[inconsistencias_df['indice'].unique()].sample(5)
#
#
# ============================================================================
# 24. SEGMENTACIONES
# ============================================================================
# Objetivo
# Identificar grupos con diferente comportamiento respecto a la variable objetivo (TV).

# Las segmentaciones suelen realizarse después del análisis individual,
# correlaciones y análisis bivariante.

# SEGMENTACIÓN NUMÉRICA
# Convertir una variable continua en grupos mediante pd.cut() o qcut().

##### Segmentación numérica

# Objetivo:
# - Convertir variables continuas en grupos interpretables.
# - Comparar segmentos frente a la TV.

# Herramientas:
# - pd.cut()
# - pd.qcut() --> 

# Análisis posterior:
#- groupby()
# - countplot_hue()
# - tablas de porcentajes

# Ejemplo 1
#df["edad_grupo"] = pd.cut(
   # df["edad"],
   # bins=[0,18,40,60,100],

# Ejemplo 
#df["edad_grupo"] = pd.cut(
   # df["edad"],
   # bins=[0,18,40,60,100],
   # labels=[
      #  "Niño","Adulto joven", "Adulto", "Mayor" ])

# o partes iguales. 
# pd.qcut(
  #  titanic["tarifa"],
   # q=3)
# se puede incluir labels. 

#Posteriormente analizar:
# edad_grupo vs TV
# pd.crosstab(
   # pedidos["edad_grupo"],
   # pedidos[tv2],
   # normalize="index").round(3) * 100

# SEGMENTACIÓN CATEGÓRICA
#Analizar categorías existentes frente a la variable objetivo.
# Objetivo:
# Identificar qué categorías presentan mejor o peor comportamiento.

# SEGMENTACIÓN COMBINADA
# Combinar dos o más variables para detectar patrones más específicos.
# Objetivo:
# Identificar perfiles concretos de interés.
