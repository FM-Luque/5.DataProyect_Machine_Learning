"""
sp_clustering.py

Caja de herramientas de Clustering.

Sigue el flujo del temario:

1. PREPARACIÓN
    estandarizar_clustering()

2. ELECCIÓN DEL NÚMERO DE CLUSTERS
    metodo_codo()
    metodo_silhouette()

3. ENTRENAMIENTO
    entrenar_kmeans()

4. EVALUACIÓN
    evaluar_clustering()

5. VISUALIZACIÓN
    visualizar_clusters_pca()

6. INTERPRETACIÓN
    perfil_clusters()

Los modelos (KMeans, DBSCAN, AgglomerativeClustering)
pueden seguir utilizándose directamente desde sklearn.

Este módulo encapsula únicamente tareas repetitivas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

from sklearn.decomposition import PCA

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 2000)


# ==========================================================================
# 1. PREPARACIÓN
# ==========================================================================

def estandarizar_clustering(
    df,
    columnas=None
):
    """
    Estandariza variables numéricas.
    """

    scaler = StandardScaler()

    df = df.copy()

    if columnas is None:

        columnas = df.select_dtypes(
            include=np.number
        ).columns

    df[columnas] = scaler.fit_transform(
        df[columnas]
    )

    return df, scaler


# ==========================================================================
# 2. ELEGIR NÚMERO DE CLUSTERS
# ==========================================================================

def metodo_codo(
    X,
    k_max=10,
    random_state=42
):
    """
    Elbow Method.
    """

    inercias = []

    ks = range(1, k_max + 1)

    for k in ks:

        modelo = KMeans(
            n_clusters=k,
            random_state=random_state
        )

        modelo.fit(X)

        inercias.append(
            modelo.inertia_
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        ks,
        inercias,
        marker="o"
    )

    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.title("Método del codo")

    plt.grid()

    plt.show()

    return inercias


def metodo_silhouette(
    X,
    k_max=10,
    random_state=42
):
    """
    Busca el mejor k mediante silhouette.
    """

    scores = []

    ks = range(2, k_max + 1)

    for k in ks:

        modelo = KMeans(
            n_clusters=k,
            random_state=random_state
        )

        labels = modelo.fit_predict(X)

        score = silhouette_score(
            X,
            labels
        )

        scores.append(score)

    mejor_k = ks[np.argmax(scores)]

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(ks),
        scores,
        marker="o"
    )

    plt.xlabel("Número de clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Método Silhouette")

    plt.grid()

    plt.show()

    print(f"Mejor k: {mejor_k}")

    return mejor_k, max(scores)


# ==========================================================================
# 3. ENTRENAMIENTO
# ==========================================================================

def entrenar_kmeans(
    X,
    n_clusters=3,
    random_state=42
):
    """
    Entrena KMeans.
    """

    modelo = KMeans(
        n_clusters=n_clusters,
        random_state=random_state
    )

    labels = modelo.fit_predict(X)

    return modelo, labels


# ==========================================================================
# 4. EVALUACIÓN
# ==========================================================================

def evaluar_clustering(
    X,
    labels,
    modelo=None
):
    """
    Evalúa clustering.
    """

    resultados = {}

    if len(np.unique(labels)) > 1:

        resultados["Silhouette"] = (
            silhouette_score(
                X,
                labels
            )
        )

        resultados["Davies_Bouldin"] = (
            davies_bouldin_score(
                X,
                labels
            )
        )

    if (
        modelo is not None and
        hasattr(modelo, "inertia_")
    ):

        resultados["Inercia"] = (
            modelo.inertia_
        )

    return resultados


# ==========================================================================
# 5. VISUALIZACIÓN
# ==========================================================================

def visualizar_clusters_pca(
    X,
    labels
):
    """
    PCA + Scatter.
    """

    pca = PCA(
        n_components=2
    )

    componentes = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        componentes[:, 0],
        componentes[:, 1],
        c=labels,
        cmap="viridis"
    )

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.title(
        "Clusters representados con PCA"
    )

    plt.colorbar(scatter)

    plt.show()


# ==========================================================================
# 6. INTERPRETACIÓN
# ==========================================================================

def perfil_clusters(
    df,
    labels,
    columnas=None
):
    """
    Perfil de clusters.
    """

    datos = df.copy()

    datos["Cluster"] = labels

    if columnas is None:

        columnas = datos.select_dtypes(
            include=np.number
        ).columns

        columnas = [
            col for col in columnas
            if col != "Cluster"
        ]

    resumen = (
        datos
        .groupby("Cluster")[columnas]
        .mean()
        .round(2)
    )

    return resumen


# ==========================================================================
# EJEMPLO DE USO (solo se ejecuta si corres este archivo directamente,
# NUNCA al hacer 'from sp_clustering import *' desde un notebook)
# ==========================================================================

if __name__ == "__main__":

    # Este bloque es solo una plantilla de referencia — sustituye X
    # por tu propio DataFrame antes de ejecutar este archivo suelto.

    # Escalado
    X_scaled, scaler = estandarizar_clustering(X)

    # Encontrar k
    mejor_k, score = metodo_silhouette(X_scaled)

    # Entrenar
    modelo, labels = entrenar_kmeans(
        X_scaled,
        n_clusters=mejor_k
    )

    # Evaluar
    print(
        evaluar_clustering(
            X_scaled,
            labels,
            modelo
        )
    )

    # Visualizar
    visualizar_clusters_pca(
        X_scaled,
        labels
    )

    # Interpretar
    print(
        perfil_clusters(
            X,
            labels
        )
    )