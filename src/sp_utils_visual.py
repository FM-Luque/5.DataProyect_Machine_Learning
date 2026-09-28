"""
sp_utils_visual.py
================
Cuaderno de referencia (bitácora) — igual que sp_utils.py, NO es un
módulo con funciones para importar. Es una chuleta de comandos sueltos
de matplotlib y seaborn para visualización rápida en EDA.

Cómo usarlo: busca la sección que necesites, copia el código y pégalo
en tu notebook, cambiando "df" y los nombres de columna por los tuyos.

Todo está comentado con "#" a propósito, para que el archivo no dé
error si lo abres o ejecutas por accidente.
"""


# ============================================================================
# 0. CONFIGURACIÓN INICIAL — lo primero de cualquier notebook con gráficos
# ============================================================================

# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(8, 5))
#   Abre una figura nueva con un tamaño concreto (ancho, alto en
#   pulgadas). Ponlo ANTES de cada gráfico si quieres controlar el
#   tamaño; si no, usa el tamaño por defecto (bastante pequeño).

# sns.set_style('whitegrid')
#   Estilo de fondo para todos los gráficos de seaborn (rejilla suave
#   en vez de fondo gris). Se pone una vez al principio del notebook.

# plt.show()
#   Muestra el gráfico. En Jupyter normalmente no hace falta (se ve
#   solo), pero conviene ponerlo si generas varios gráficos en la
#   misma celda para que no se mezclen entre sí.


# ============================================================================
# 1. UNA VARIABLE NUMÉRICA — ver su distribución
# ============================================================================

# sns.histplot(df['edad'], bins=30)
#   Histograma: agrupa los valores en "cajas" (bins) y cuenta cuántas
#   filas caen en cada una. Lo primero para ver la forma de una
#   variable numérica (¿está centrada? ¿tiene cola larga?).

# sns.histplot(df['edad'], bins=30, kde=True)
#   Igual, pero añade una línea suave (KDE) por encima que estima la
#   forma de la distribución sin depender tanto del nº de bins.

# sns.boxplot(x=df['edad'])
#   Diagrama de caja: muestra mediana, cuartiles, y los outliers como
#   puntos sueltos fuera de los "bigotes". El gráfico de referencia
#   para outliers (el mismo criterio que usa tu detectar_outliers_iqr()).

# df['edad'].plot(kind='hist', bins=30)
#   La misma idea que histplot(), pero usando pandas directamente en
#   vez de seaborn (más rápido de escribir si no necesitas el KDE).


# ============================================================================
# 2. UNA VARIABLE CATEGÓRICA — ver cuántos hay de cada tipo
# ============================================================================

# sns.countplot(x='ocupacion', data=df)
#   Barras con el nº de filas de cada categoría. El equivalente
#   visual de tu value_counts().

# sns.countplot(y='ocupacion', data=df)
#   Igual, pero en horizontal — mucho más legible cuando hay muchas
#   categorías o los nombres son largos (evita que se solapen).

# df['ocupacion'].value_counts().plot(kind='bar')
#   Misma idea con pandas directo. Ojo: por defecto ordena de más a
#   menos frecuente, countplot() de seaborn no lo hace solo.

# sns.countplot(x='ocupacion', data=df, order=df['ocupacion'].value_counts().index)
#   countplot() de seaborn, pero forzando el orden de más a menos
#   frecuente (el "order=" es la clave para conseguirlo).


# ============================================================================
# 3. DOS VARIABLES NUMÉRICAS — ver si están relacionadas
# ============================================================================

# sns.scatterplot(x='edad', y='ingresos', data=df)
#   Nube de puntos: cada fila es un punto. El gráfico básico para ver
#   si dos variables numéricas se mueven juntas (relación lineal,
#   agrupamientos, outliers conjuntos).

# sns.scatterplot(x='edad', y='ingresos', hue='objetivo', data=df)
#   Igual, pero coloreando los puntos según una tercera columna
#   categórica (hue = "matiz"). Muy usado para ver si la relación
#   cambia según el grupo (ej. clientes que sí/no contrataron).

# sns.regplot(x='edad', y='ingresos', data=df)
#   Como scatterplot(), pero añade una línea de tendencia (regresión
#   lineal simple) por encima. Útil para ver de un vistazo si la
#   relación es creciente, decreciente o plana.


# ============================================================================
# 4. UNA CATEGÓRICA + UNA NUMÉRICA — comparar un número entre grupos
# ============================================================================

# sns.boxplot(x='ocupacion', y='ingresos', data=df)
#   Un boxplot por cada categoría, todos juntos. El gráfico de
#   referencia en A/B testing: comparar la distribución de una
#   variable numérica entre grupos (ej. ingresos por ocupación).

# sns.boxplot(x='objetivo', y='edad', data=df)
#   Mismo patrón aplicado a tu variable objetivo: ¿la edad se
#   distribuye distinto entre quien dijo 'yes' y quien dijo 'no'?

# sns.violinplot(x='ocupacion', y='ingresos', data=df)
#   Como el boxplot, pero mostrando también la forma completa de la
#   distribución (más ancho donde hay más datos). Da más detalle que
#   el boxplot, pero es algo más difícil de leer para alguien que
#   empieza — si tienes dudas, quédate con boxplot().

# sns.barplot(x='ocupacion', y='ingresos', data=df)
#   Barras con la MEDIA de 'ingresos' por categoría (por defecto), con
#   una pequeña línea de margen de error. Distinto del countplot(),
#   que cuenta filas: barplot() promedia una columna numérica.


# ============================================================================
# 5. DOS VARIABLES CATEGÓRICAS — comparar categorías entre sí
# ============================================================================

# sns.countplot(x='ocupacion', hue='objetivo', data=df)
#   Barras agrupadas: cuenta filas de 'ocupacion', separadas por color
#   según 'objetivo'. El equivalente visual de tu pd.crosstab().

# tabla = pd.crosstab(df['ocupacion'], df['objetivo'], normalize='index')
# sns.heatmap(tabla, annot=True, fmt='.2f', cmap='Blues')
#   Primero la tabla cruzada en %, luego un mapa de calor: cuanto más
#   oscura la celda, mayor el valor. "annot=True" escribe el número
#   dentro de cada celda, "fmt='.2f'" con 2 decimales.


# ============================================================================
# 6. MATRIZ DE CORRELACIÓN — relación entre TODAS las variables numéricas
# ============================================================================

# corr = df.corr(numeric_only=True)
# sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
#   El mapa de calor más usado en EDA: de un vistazo ves qué pares de
#   variables numéricas están más relacionadas (colores intensos) y
#   cuáles no (colores cerca del centro). "center=0" hace que el 0
#   quede en un color neutro, separando bien positivo de negativo.


# ============================================================================
# 7. EVOLUCIÓN EN EL TIEMPO — series temporales
# ============================================================================

# sns.lineplot(x='fecha', y='ingresos', data=df)
#   Línea que conecta los puntos en orden de fecha. El gráfico básico
#   para ver tendencias, subidas/bajadas o estacionalidad.

# df.groupby('fecha')['ingresos'].mean().plot()
#   Alternativa con pandas puro: agrupa por fecha (media diaria, por
#   ejemplo) y luego lo dibuja directamente con .plot(), sin seaborn.


# ============================================================================
# 8. VARIOS GRÁFICOS A LA VEZ
# ============================================================================

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# sns.histplot(df['edad'], ax=axes[0])
# sns.boxplot(x=df['edad'], ax=axes[1])
# plt.show()
#   Dos gráficos lado a lado en la misma figura. "axes[0]"/"axes[1]"
#   son cada uno de los huecos; el parámetro "ax=" le dice a cada
#   gráfico en qué hueco dibujarse. Cambia el primer "1, 2" por
#   "2, 2" si quieres una rejilla de 4 gráficos, etc.

# sns.pairplot(df[['edad', 'ingresos', 'duracion']])
#   Genera automáticamente TODOS los scatterplots posibles entre las
#   columnas indicadas (y un histograma en la diagonal). Muy cómodo
#   para una primera exploración, pero lento si le pasas muchas
#   columnas o el dataset es grande — mejor seleccionar pocas.

# sns.pairplot(df[['edad', 'ingresos', 'objetivo']], hue='objetivo')
#   Igual, coloreando por una columna categórica — para ver de golpe
#   si varias variables numéricas separan bien a los dos grupos.


# ============================================================================
# 9. DETALLES ÚTILES — títulos, etiquetas, guardar
# ============================================================================

# plt.title('Ingresos por ocupación')
#   Título del gráfico. Ponlo DESPUÉS de la línea que dibuja el
#   gráfico (sns.boxplot(...), etc.), no antes.

# plt.xlabel('Ocupación')
# plt.ylabel('Ingresos (€)')
#   Nombres de los ejes, por si el nombre de la columna no es claro
#   por sí solo (ej. 'ipc' -> 'Índice de Precios al Consumo').

# plt.xticks(rotation=45)
#   Rota las etiquetas del eje X 45 grados. Imprescindible cuando hay
#   muchas categorías con nombres largos que se solapan.

# plt.tight_layout()
#   Ajusta automáticamente los márgenes para que no se corten
#   etiquetas o títulos. Ponlo justo antes de plt.show() cuando algo
#   se vea recortado.

# plt.savefig('grafico.png', dpi=150, bbox_inches='tight')
#   Guarda el gráfico como imagen. "dpi=150" es buena calidad sin
#   pesar demasiado; "bbox_inches='tight'" evita que se corten los
#   bordes. Ponlo ANTES de plt.show() (después, la figura ya está
#   "vacía" y se guardaría en blanco).
