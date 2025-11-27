import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from mpl_toolkits.mplot3d import Axes3D

# Définir les couleurs (bleu, rouge, vert, noir, violet, cyan, etc.)
couleurs_vives = [
    "#0000FF",  # bleu
    "#800080",  # violet
    "#F76008",  # orange
    "#00AAAA",  # cyan
    "#FF00FF",  # magenta
    "#00AA00",  # vert
    "#000000",  # noir
    "#FF0000",  # rouge
]

# Créer une colormap discrète
cmap_vives = ListedColormap(couleurs_vives)


def preprocessing(
    df: pd.DataFrame, years: list, countries: list, cols: list, features: list
) -> pd.DataFrame:
    """select data and features to process and log-transform floats"""
    dfl = df.copy()
    dfl = dfl[dfl["year"].isin(years)]
    dfl = dfl[dfl["country"].isin(countries)]
    dfl = dfl[cols + features]
    dfl.reset_index(drop=True, inplace=True)

    # columns to log-transform (floats) :
    features = dfl.select_dtypes(include=["float64"]).columns.tolist()
    # floats = dfl.dtypes[df.dtypes == 'float64'].index.tolist() # same result

    # log transform because asyetrical distributions
    for feat in features:
        dfl[feat] = np.log1p(dfl[feat])  # log1p to handle zero values safely

    # standardisation :
    scaler = StandardScaler()
    X = scaler.fit_transform(dfl[features])

    return X


# test preprocessing:
# X = preprocessing(df, years, countries, cols, features)
# print(X.shape)
# X


def kmeans_training(X, k, seed):
    """kmeans training : dataset X, k clusters"""
    clusterer = KMeans(n_clusters=k, tol=1e-4, random_state=seed)
    cluster_labels = clusterer.fit_predict(X)
    centers = clusterer.cluster_centers_
    inertia = clusterer.inertia_
    return (cluster_labels, centers, inertia)


def dbscan_training(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    return cluster_labels


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


def display_clusters(X, k, features, centers, cluster_labels, pop):
    """display 2D views (feature1 x feature2) showing k centers and k clusters, for all features"""
    k = centers.shape[0]
    n_graph = len(features) * (len(features) - 1) // 2
    fig, ax = plt.subplots(
        (n_graph + 2) // 3, 3, figsize=(15, 5 * ((n_graph + 2) // 3)), squeeze=False
    )

    i = 0
    if k <= 8:
        cmap = cmap_vives
    else:
        cmap = plt.cm.get_cmap("viridis", k)

    cluster_colors = [couleurs_vives[i] for i in cluster_labels]

    for ix, feature_x in enumerate(features):
        for ixy, feature_y in enumerate(features[ix + 1 :]):
            iy = ix + 1 + ixy
            # colors = cm.nipy_spectral(cluster_labels.astype(float) / k)

            ax[i // 3, i % 3].scatter(
                X[:, ix],
                X[:, iy],
                marker="o",
                s=pop / 1e6,  # dots de taille proportionnelle à la population
                lw=0,
                alpha=0.7,
                c=cluster_colors,
                edgecolor="k",
            )
            if k != 0:
                # On affiche le numéro du cluster à son centre
                # On dessine un cercle blanc autour des points pour les faire ressortir
                ax[i // 3, i % 3].scatter(
                    centers[:, ix],
                    centers[:, iy],
                    marker="o",
                    c="white",
                    alpha=1,
                    s=200,
                    edgecolor="k",
                )

                for ic, c in enumerate(centers):
                    ax[i // 3, i % 3].scatter(
                        c[ix], c[iy], marker="$%d$" % ic, alpha=1, s=50, edgecolor="k"
                    )

            ax[i // 3, i % 3].set_xlabel(f"{features[ix]}, [{ix}]")
            ax[i // 3, i % 3].set_ylabel(f"{features[iy]}, [{iy}]")

            plt.suptitle(
                "Visualisation des clusters sur les données avec n_clusters = %d" % k,
                fontsize=14,
                fontweight="bold",
            )
            i += 1
    return fig


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


def display_3D(X, cluster_labels, pop, iso_code):
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111, projection="3d")
    cluster_colors = [couleurs_vives[i] for i in cluster_labels]
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_colors, s=pop / 1e6, alpha=0.7)

    for xi, yi, zi, label in zip(X[:, 0], X[:, 1], X[:, 2], iso_code):
        ax.text(
            xi + 0.01, yi + 0.01, zi + 0.01, label, fontsize=8, ha="left", va="center"
        )

    ax.set_xlabel("co2_per_energy (scaled)")
    ax.set_ylabel("energy_per_gdp (scaled)")
    ax.set_zlabel("gdp_per_capita (scaled)")
    ax.set_title("Spectral Clustering (RFM)")
    return fig
