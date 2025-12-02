import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from utils.display import couleurs_vives


def display_silhouettes(
    X, k, cluster_labels, eps=None, min_samples=None, inertia=None, bw=None, gamma=None
):
    """display silhouette charts for each of the k clusters"""
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)

    ax_height = len(X) + (k + 1) * 10
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, ax_height])

    if k > 0 and k < X.shape[0]:
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_sample_values = silhouette_samples(X, cluster_labels)
    else:
        silhouette_avg = 0
        silhouette_sample_values = np.zeros(cluster_labels.shape)

    # if k <= cmap_vives.N:
    #     cmap = cmap_vives
    # else:
    #     cmap = plt.cm.get_cmap("viridis", k)

    y_lower = 10
    for i in range(k):
        # On trie les valeurs de silhouette pour chaque cluster
        ith_cluster_silhouette_values = silhouette_sample_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        sil_avg_i = np.mean(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i

        # color = cm.nipy_spectral(float(i) / k)

        color = couleurs_vives[i]

        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax.text(
            -0.05,
            y_lower + 0.5 * size_cluster_i,
            str(i),
            fontweight="bold",
            bbox=dict(boxstyle="Circle", facecolor="none"),
        )
        ax.text(0.10, y_lower + 0.5 * size_cluster_i, f"n={size_cluster_i}")
        ax.text(0.8, y_lower + 0.5 * size_cluster_i, f"sil={sil_avg_i:.3f}")

        # On calcule la nouvelle valeur de y_lower pour le prochain plot
        y_lower = y_upper + 10

    ax.text(0.10, ax_height * 0.99, f"outliers={(cluster_labels==-1).sum()}", va="top")
    ax.text(
        -0.09,
        ax_height * 0.99,
        f"k={k}",
        va="top",
        fontweight="bold",
    )
    legend = ""
    if eps:
        legend += f"epsillon = {eps}\n"
    if min_samples:
        legend += f"min_samples = {min_samples}\n"
    if inertia:
        legend += f"inertie = {inertia:.0f}"
    if bw:
        legend += f"bandwidth = {bw}"
    if gamma:
        legend += f"gamma = {gamma}"
    ax.text(
        0.80,
        ax_height * 0.99,
        str(legend),
        va="top",
        fontweight="bold",
    )
    ax.set_title(
        f"Coef de silhouette moyen = {silhouette_avg:.3f}",
        color="red",
        fontsize=12,
    )
    ax.set_xlabel("Valeurs des coefficients de silhouette")
    ax.set_ylabel("Label du cluster")

    # La ligne rouge est la moyenne de tous les coefficients de silhouette
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    return fig, ax


def display_scores(range_n_clusters, silhouette_scores, inertia_scores):
    """display silhouette and inertia scores as a function of k (number of clusters)"""

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    best_k = range_n_clusters[int(np.argmax(silhouette_scores))]

    ax[0].plot(range_n_clusters, silhouette_scores, marker="o")
    ax[0].set_title("Silhouette Score vs Number of Clusters")
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("Silhouette Score")
    ax[0].axvline(best_k, color="red", linestyle="--", label=f"best k = {best_k}")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(range_n_clusters, inertia_scores, marker="o", color="orange")
    ax[1].set_title("Inertia vs Number of Clusters")
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Inertia")
    ax[1].grid(alpha=0.3)

    plt.suptitle("KMeans Clustering Evaluation Metrics", fontsize=16, fontweight="bold")

    return fig


def display_scores_ms(bw_values, silhouette_scores, estimate_bw):
    """display silhouette and inertia scores as a function of k (number of clusters)"""

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    best_bw = bw_values[int(np.argmax(silhouette_scores))]

    ax.plot(bw_values, silhouette_scores, marker="o")
    ax.set_title("Silhouette Score vs Bandwidth")
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Silhouette Score")
    ax.axvline(
        best_bw,
        color="red",
        linestyle="--",
        label=f"best bw = {best_bw}\nestimate_bw = {estimate_bw}",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        "MeanShift Clustering Evaluation Metrics", fontsize=16, fontweight="bold"
    )

    return fig
