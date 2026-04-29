import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

CLUSTER_COLOURS = {
    'tank':   '#D94040',
    'flying': '#4060D9',
    'smart':  '#40B040',
}

SPECIES_COLOURS = {
    'orc':      '#C03030',
    'lizard':   '#E87030',
    'wingedrat':'#3050D0',
    'human':    '#30A030',
    'halfling': '#60D060',
}

SPECIES_MARKERS = {
    'orc':      's',
    'lizard':   '^',
    'wingedrat':'D',
    'human':    'o',
    'halfling': 'P',
}


# ─── K-Means cluster count selection ─────────────────────────────────────────

def plot_kmeans_selection(X_pca, k_range=range(2, 9), n_chosen=3, save_dir=None):
    """
    Elbow curve (WCSS inertia) and silhouette score vs k.
    Both metrics should agree on the optimal cluster count.
    """
    inertias    = []
    silhouettes = []
    ks          = list(k_range)

    for k in ks:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ks, inertias, marker='o', color='steelblue', linewidth=2)
    axes[0].axvline(x=n_chosen, color='red', linestyle='--', alpha=0.7,
                    label=f'k={n_chosen} chosen')
    axes[0].set_xlabel("k (number of clusters)")
    axes[0].set_ylabel("Inertia (WCSS)")
    axes[0].set_title("K-Means — Elbow Curve", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, silhouettes, marker='o', color='tomato', linewidth=2)
    axes[1].axvline(x=n_chosen, color='red', linestyle='--', alpha=0.7,
                    label=f'k={n_chosen} chosen')
    axes[1].set_xlabel("k (number of clusters)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("K-Means — Silhouette Score", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("K-Means: Cluster Count Selection", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "kmeans_selection.png"), dpi=150)
    plt.close()
    print("Saved: kmeans_selection.png")


# ─── GMM model selection (BIC / AIC) ─────────────────────────────────────────

def plot_gmm_selection(X_pca, k_range=range(2, 9), n_chosen=3, save_dir=None):
    """
    BIC and AIC vs n_components for GMM.
    The minimum of BIC indicates the statistically optimal component count.
    """
    bics = []
    aics = []
    ks   = list(k_range)

    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm.fit(X_pca)
        bics.append(gmm.bic(X_pca))
        aics.append(gmm.aic(X_pca))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, bics, marker='o', label='BIC', color='steelblue', linewidth=2)
    ax.plot(ks, aics, marker='s', label='AIC', color='tomato',    linewidth=2)
    ax.axvline(x=n_chosen, color='grey', linestyle='--', alpha=0.8,
               label=f'k={n_chosen} chosen')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Score (lower = better)")
    ax.set_title("GMM — BIC / AIC Model Selection\n"
                 "Minimum indicates optimal cluster count",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "gmm_selection.png"), dpi=150)
    plt.close()
    print("Saved: gmm_selection.png")


# ─── Named cluster scatter ─────────────────────────────────────────────────────

def plot_named_clusters(X_pca_2d, cluster_labels, cluster_names,
                        species=None, save_dir=None):
    """
    PCA 2D scatter coloured by named cluster (tank/flying/smart).
    If species list is provided, each species uses a distinct marker shape,
    showing that the clusters cleanly separate by behaviour type.
    """
    named_labels = [cluster_names[c] for c in cluster_labels]

    fig, axes = plt.subplots(1, 2 if species is not None else 1,
                             figsize=(14 if species is not None else 8, 6))
    if species is None:
        axes = [axes]

    # Left: colour by cluster
    for name in cluster_names:
        mask = np.array(named_labels) == name
        col  = CLUSTER_COLOURS.get(name, 'grey')
        axes[0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                        c=col, alpha=0.6, s=50, label=name)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title("GMM Clusters — named", fontweight='bold')
    axes[0].legend(title="Cluster", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Right: colour by cluster, shape by species
    if species is not None:
        sp_arr = np.array(species)
        for sp in np.unique(sp_arr):
            mask   = sp_arr == sp
            cols   = [CLUSTER_COLOURS.get(named_labels[i], 'grey')
                      for i in np.where(mask)[0]]
            marker = SPECIES_MARKERS.get(sp, 'o')
            axes[1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                            c=cols, marker=marker, alpha=0.7, s=60, label=sp)
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
        axes[1].set_title("GMM Clusters — shape=species, colour=cluster",
                          fontweight='bold')
        # Legend: species (shapes)
        shape_handles = [mpatches.Patch(color='grey', label='shape = species')]
        for sp, mk in SPECIES_MARKERS.items():
            if sp in np.unique(sp_arr):
                shape_handles.append(
                    plt.Line2D([0], [0], marker=mk, color='grey',
                               linestyle='None', markersize=8, label=sp))
        axes[1].legend(handles=shape_handles, fontsize=8, ncol=2)
        axes[1].grid(True, alpha=0.3)

        # Shared cluster colour legend
        cluster_handles = [mpatches.Patch(color=CLUSTER_COLOURS[n], label=n)
                           for n in cluster_names if n in CLUSTER_COLOURS]
        fig.legend(handles=cluster_handles, title="Cluster", loc='lower center',
                   ncol=len(cluster_names), fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Behaviour Clusters in PCA Space", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "named_clusters.png"),
                    dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: named_clusters.png")


# ─── Cluster radar chart ──────────────────────────────────────────────────────

def plot_cluster_radar(cluster_profiles: pd.DataFrame, cluster_names: list,
                       save_dir=None):
    """
    Radar / spider chart of normalised mean feature values per cluster.
    The shape of each polygon directly explains the combat type:
      tank   → spiky on strength/stench axes
      flying → spiky on flight axis
      smart  → spiky on intelligence axis
    """
    features = cluster_profiles.columns.tolist()
    profiles  = cluster_profiles.values.astype(float)

    # Normalise each feature to [0, 1] across clusters
    mins   = profiles.min(axis=0)
    maxs   = profiles.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    profiles_norm = (profiles - mins) / ranges

    n_feat = len(features)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, cname in enumerate(cluster_names):
        vals = profiles_norm[i].tolist() + [profiles_norm[i][0]]
        col  = CLUSTER_COLOURS.get(cname, f'C{i}')
        ax.plot(angles, vals, 'o-', linewidth=2.5, label=cname, color=col)
        ax.fill(angles, vals, alpha=0.12, color=col)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Cluster Feature Profiles (normalised)\n"
                 "Justifies combat-type mapping to each cluster",
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "cluster_radar.png"),
                    dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_radar.png")


# ─── Species composition per cluster ─────────────────────────────────────────

def plot_species_composition(df: pd.DataFrame, cluster_labels,
                             cluster_names: list, save_dir=None):
    """
    Stacked bar chart: which species end up in each named cluster.
    Validates the cluster→combat type mapping:
      tank   should be almost entirely orc + lizard
      flying should be almost entirely wingedrat
      smart  should be almost entirely human + halfling
    """
    df_tmp = df.copy()
    df_tmp['cluster_name'] = [cluster_names[c] for c in cluster_labels]

    comp     = pd.crosstab(df_tmp['cluster_name'], df_tmp['species'])
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100

    species_present = comp.columns.tolist()
    colours         = [SPECIES_COLOURS.get(s, 'grey') for s in species_present]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    comp.plot(kind='bar', ax=axes[0], color=colours, edgecolor='white', width=0.6)
    axes[0].set_title("Species Count per Cluster", fontweight='bold')
    axes[0].set_xlabel("Cluster"); axes[0].set_ylabel("Count")
    axes[0].legend(title="Species", fontsize=9)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0, fontsize=11)
    axes[0].grid(True, axis='y', alpha=0.3)

    comp_pct.plot(kind='bar', stacked=True, ax=axes[1],
                  color=colours, edgecolor='white', width=0.6)
    axes[1].set_title("Species Composition per Cluster (%)", fontweight='bold')
    axes[1].set_xlabel("Cluster"); axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(title="Species", fontsize=9)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0, fontsize=11)
    axes[1].set_ylim(0, 115)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle("Species Membership per Behaviour Cluster\n"
                 "Validates cluster → combat type mapping",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "species_composition.png"), dpi=150)
    plt.close()
    print("Saved: species_composition.png")


# ─── PCA loadings ─────────────────────────────────────────────────────────────

def plot_pca_loadings(pca_2d, feature_cols: list, save_dir=None):
    """
    Heatmap and grouped bar chart of feature contributions to PC1 and PC2.
    Shows which sensor features drive the separation between clusters.
    """
    n_pcs    = len(pca_2d.components_)
    loadings = pd.DataFrame(
        pca_2d.components_,
        columns=feature_cols,
        index=[f"PC{i+1}" for i in range(n_pcs)],
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=axes[0], cbar_kws={'label': 'Loading'})
    axes[0].set_title("PCA Loadings Heatmap\n(feature contribution per component)",
                      fontweight='bold')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=40, ha='right')

    x = np.arange(len(feature_cols))
    w = 0.35
    axes[1].bar(x - w/2, loadings.loc['PC1'], w, label='PC1',
                color='steelblue', alpha=0.85)
    axes[1].bar(x + w/2, loadings.loc['PC2'], w, label='PC2',
                color='tomato',    alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_cols, rotation=40, ha='right', fontsize=9)
    axes[1].set_ylabel("Loading")
    axes[1].set_title("Feature Contributions to PC1 and PC2", fontweight='bold')
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_loadings.png"), dpi=150)
    plt.close()
    print("Saved: pca_loadings.png")


# ─── GMM confidence distribution ─────────────────────────────────────────────

def plot_gmm_confidence(probs: np.ndarray, cluster_names: list, save_dir=None):
    """
    Histogram of each entity's maximum cluster probability.
    Values near 1.0 mean the GMM is highly confident → well-separated clusters.
    Split by cluster to see which combat type has cleaner separation.
    """
    max_probs = probs.max(axis=1)
    dominant  = probs.argmax(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(max_probs, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(max_probs.mean(), color='red', linestyle='--',
                    label=f'mean = {max_probs.mean():.2f}')
    axes[0].set_xlabel("Max Cluster Probability")
    axes[0].set_ylabel("Count")
    axes[0].set_title("GMM Assignment Confidence\n(overall)", fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for i, name in enumerate(cluster_names):
        mask = dominant == i
        col  = CLUSTER_COLOURS.get(name, f'C{i}')
        axes[1].hist(max_probs[mask], bins=20, label=name, color=col,
                     alpha=0.6, edgecolor='white')
    axes[1].set_xlabel("Max Cluster Probability")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Distribution per Cluster", fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)

    fig.suptitle("GMM Cluster Assignment Confidence",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "gmm_confidence.png"), dpi=150)
    plt.close()
    print("Saved: gmm_confidence.png")
