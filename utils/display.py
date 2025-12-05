import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


units = {
    "co2": "Mt/year",
    "energy": "kWh/year",
    "gdp": "$/year",
    "population": "",
    "co2_per_unit_energy": "kg/kWh",
    "energy_per_gdp": "kWh/$(2011)",
    "gdp_per_capita": "$(2011)/capita/year",
}


couleurs_vives = [
    "#E53935",
    "#9C27B0",
    "#A7C7E7",
    "#B2DFDB",
    "#D87C5B",
    "#00AAAA",  # cyan
    "#00AA00",  # vert
    "#F76008",  # orange
    "#0000FF",  # bleu
    "#800080",  # violet
    "#888888",
    "#999900",
]

color_map = dict(zip(range(5), couleurs_vives))


# Créer une colormap discrète
cmap_vives = ListedColormap(couleurs_vives)


def cluster_color(df):
    if "cluster" in df.columns:
        df["cluster_color"] = [couleurs_vives[i] for i in df["cluster"]]
    return df


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
                # s=30,
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
            plt.text(
                x=0.5,
                y=0.90,
                s="Surface des points proportionnelle à la population",
                fontsize=12,
                ha="center",
                transform=fig.transFigure,
            )
            i += 1
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
