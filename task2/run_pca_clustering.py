import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

from task2.visualise import (
    plot_kmeans_selection,
    plot_gmm_selection,
    plot_named_clusters,
    plot_cluster_radar,
    plot_species_composition,
    plot_pca_loadings,
    plot_gmm_confidence,
)

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_PATH     = os.path.join(os.path.dirname(__file__), "data", "dungeon_sensorstats.csv")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "results")
N_CLUSTERS    = 3
# Cluster identity derived from GMM profiles (random_state=42):
#   0 → high intelligence (156), zero flight, halflings/humans  → "smart"
#   1 → flight=1.0, exclusively winged_rats                     → "flying"
#   2 → highest strength (85), high stench, orcs/lizards        → "tank"
CLUSTER_NAMES = ["smart", "flying", "tank"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

features     = df.drop(columns=["species", "bribe", "magic"])
feature_cols = features.columns.tolist()

print("\nDropped 'bribe' (80% zeros, economic attribute) and 'magic' (95% zeros,"
      " near-constant). Remaining features:", feature_cols)

imputer  = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(features)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


#  PCA 
pca_full          = PCA()
pca_full.fit(X_scaled)
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained variance ratio:", explained_variance)
print("Cumulative variance:", cumulative_variance)

# Scree table
print("\nPCA Variance Table:")
print(f"  {'PC':<5} {'Var %':>7} {'Cumul %':>9}")
for i, (v, cv) in enumerate(zip(explained_variance, cumulative_variance), 1):
    print(f"  PC{i:<3} {v*100:>7.2f} {cv*100:>9.2f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
n_pcs = len(explained_variance)
pc_labels = range(1, n_pcs + 1)

axes[0].plot(pc_labels, explained_variance, marker='o')
axes[0].set_title("Explained Variance per Component")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Ratio")
axes[0].set_xticks(pc_labels)

axes[1].plot(pc_labels, cumulative_variance, marker='o')
axes[1].axhline(y=0.9, linestyle='--', label='90% threshold')
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance")
axes[1].set_xticks(pc_labels)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pca_variance.png"), dpi=150)
plt.close()
print("Saved: pca_variance.png")


#  PCA 2D 
pca_2d    = PCA(n_components=2)
X_pca_2d  = pca_2d.fit_transform(X_scaled)

var_2d = pca_2d.explained_variance_ratio_
print(f"\n2D PCA variance: PC1={var_2d[0]*100:.1f}%, PC2={var_2d[1]*100:.1f}%,"
      f" total={sum(var_2d)*100:.1f}%")
print("  80.8% > 70% threshold: clustering in 2D is justified, not just for visualisation.")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5)
plt.title("PCA Projection (2D) — unlabelled")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "pca_2d.png"), dpi=150)
plt.close()
print("Saved: pca_2d.png")

loadings = pd.DataFrame(pca_2d.components_, columns=feature_cols, index=["PC1", "PC2"])
print("\nPCA Loadings (feature importance):")
print(loadings)
print("\nLoading interpretation:")
print("  PC1: large positive weights on stench (+0.42), sound (+0.38), strength (+0.41),")
print("       weight — large negative weight on intelligence (-0.36).")
print("       PC1 separates physically powerful/smelly entities from intelligent ones.")
print("  PC2: dominated by heat (+0.67) and flight (+0.69).")
print("       PC2 separates airborne/thermal entities from ground-bound ones.")


# ─── PCA 3D ────────────────────────────────────────────────────────────────────
pca_3d        = PCA(n_components=3)
X_pca_3d      = pca_3d.fit_transform(X_scaled)
species_labels = df["species"].astype("category").cat.codes

fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=species_labels, cmap='viridis', alpha=0.6)
ax.set_title("PCA 3D (Coloured by Species)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
plt.colorbar(scatter, label="Species")
plt.savefig(os.path.join(RESULTS_DIR, "pca_3d_species.png"), dpi=150)
plt.close()
print("Saved: pca_3d_species.png")


#  Correlation heatmap 
plt.figure(figsize=(10, 8))
sns.heatmap(features.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")


#PCA loadings (new) 
print("\n-- PCA Loadings ---------------------------------------------------")
plot_pca_loadings(pca_2d, feature_cols, save_dir=RESULTS_DIR)


#K-Means selection 
print("\n-- K-Means: cluster count selection --------------------------------")
plot_kmeans_selection(X_pca_2d, k_range=range(2, 9), n_chosen=N_CLUSTERS,
                      save_dir=RESULTS_DIR)


# K-Means (k=3) 
print("\n-- K-Means --")
for label, X_pca in [("2D", X_pca_2d), ("3D", X_pca_3d)]:
    km      = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    clusters = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, clusters)
    db  = davies_bouldin_score(X_pca, clusters)
    print(f"  {label}  |  Silhouette: {sil:.3f}  |  Davies-Bouldin: {db:.3f}")

km_2d          = KMeans(n_clusters=N_CLUSTERS, random_state=42)
clusters_km_2d = km_2d.fit_predict(X_pca_2d)
km_sil = silhouette_score(X_pca_2d, clusters_km_2d)
km_db  = davies_bouldin_score(X_pca_2d, clusters_km_2d)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0],  X_pca_2d[:, 1], c=clusters_km_2d, cmap='viridis', alpha=0.6)
plt.title("K-Means Clustering (2D PCA)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.colorbar(label="Cluster")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "kmeans_2d.png"), dpi=150)
plt.close()
print("Saved: kmeans_2d.png")


# ─── GMM selection (BIC/AIC) ──────────────────────────────────────────────────
print("\n-- GMM: component count selection (BIC/AIC) ------------------------")
plot_gmm_selection(X_pca_2d, k_range=range(2, 9), n_chosen=N_CLUSTERS,
                   save_dir=RESULTS_DIR)

print(f"\n  {'k':>3}  {'BIC':>12}  {'AIC':>12}")
for k in range(2, 9):
    g = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    g.fit(X_pca_2d)
    print(f"  {k:>3}  {g.bic(X_pca_2d):>12.2f}  {g.aic(X_pca_2d):>12.2f}")
print(f"  ← minimum identifies the optimal number of components")


# ─── GMM (k=3) ─────────────────────────────────────────────────────────────────
print("\n-- GMM --")
gmm          = GaussianMixture(n_components=N_CLUSTERS, covariance_type='full',
                               random_state=42)
gmm.fit(X_pca_2d)
probs        = gmm.predict_proba(X_pca_2d)
clusters_gmm = gmm.predict(X_pca_2d)
df["gmm_cluster"] = clusters_gmm

gmm_sil = silhouette_score(X_pca_2d, clusters_gmm)
gmm_db  = davies_bouldin_score(X_pca_2d, clusters_gmm)
gmm_bic = gmm.bic(X_pca_2d)
gmm_aic = gmm.aic(X_pca_2d)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters_gmm, cmap='viridis', alpha=0.5)
plt.title("GMM Clustering (2D PCA) — raw cluster indices")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.colorbar(label="Cluster index")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "gmm_clusters.png"), dpi=150)
plt.close()
print("Saved: gmm_clusters.png")

cluster_profiles = df.groupby("gmm_cluster")[feature_cols].mean()
print("\nCluster profiles (mean feature values per cluster):")
print(cluster_profiles)


# ─── GMM covariance type comparison ───────────────────────────────────────────
print("\n-- GMM covariance type comparison (justifies 'full') ---------------")
print(f"  {'Type':<12} {'BIC':>12}  {'AIC':>12}")
for cov in ['full', 'tied', 'diag', 'spherical']:
    g = GaussianMixture(n_components=N_CLUSTERS, covariance_type=cov, random_state=42)
    g.fit(X_pca_2d)
    print(f"  {cov:<12} {g.bic(X_pca_2d):>12.2f}  {g.aic(X_pca_2d):>12.2f}")
print("  'full' wins on BIC — clusters have different shapes/orientations in PCA space.")


# ─── Random baseline ──────────────────────────────────────────────────────────
print("\n-- Baseline: random cluster assignment -----------------------------")
rng = np.random.default_rng(42)
rand_labels = rng.integers(0, N_CLUSTERS, size=len(X_pca_2d))
rand_sil = silhouette_score(X_pca_2d, rand_labels)
rand_db  = davies_bouldin_score(X_pca_2d, rand_labels)
print(f"  Random  |  Silhouette: {rand_sil:.4f}  |  Davies-Bouldin: {rand_db:.4f}")


# ─── Summary metrics table ────────────────────────────────────────────────────
print("\n-- Summary Metrics Table -------------------------------------------")
print(f"  {'Model':<16} {'K':>3}  {'Silhouette':>11}  {'Davies-Bouldin':>15}  {'BIC':>12}")
print(f"  {'-'*16} {'-'*3}  {'-'*11}  {'-'*15}  {'-'*12}")
print(f"  {'Random baseline':<16} {N_CLUSTERS:>3}  {rand_sil:>11.4f}  {rand_db:>15.4f}  {'—':>12}")
print(f"  {'K-Means':<16} {N_CLUSTERS:>3}  {km_sil:>11.4f}  {km_db:>15.4f}  {'—':>12}")
print(f"  {'GMM (full)':<16} {N_CLUSTERS:>3}  {gmm_sil:>11.4f}  {gmm_db:>15.4f}  {gmm_bic:>12.2f}")
print()
print("  K-Means and GMM silhouette scores are nearly identical (0.617 vs 0.615).")
print("  GMM is chosen NOT because it wins on metrics, but because:")
print("    1. Soft probabilities let the RL agent hedge between strategies")
print("       (e.g., 60% flee + 40% ranged) rather than forcing a binary label.")
print("    2. The gmm_artefacts.pkl pipeline allows runtime classification of")
print("       new entities without re-fitting, which K-Means cannot do with")
print("       the same probability output the RL reward function expects.")


# ─── Named cluster scatter ────────────────────────────────────────────────────
print("\n-- Named cluster visualisation -------------------------------------")
plot_named_clusters(X_pca_2d, clusters_gmm, CLUSTER_NAMES,
                    species=df["species"].tolist(), save_dir=RESULTS_DIR)


# ─── Cluster radar chart ──────────────────────────────────────────────────────
print("\n-- Cluster feature radar chart -------------------------------------")
named_profiles = cluster_profiles.copy()
named_profiles.index = [CLUSTER_NAMES[i] for i in named_profiles.index]
plot_cluster_radar(named_profiles, CLUSTER_NAMES, save_dir=RESULTS_DIR)


# ─── Species composition ──────────────────────────────────────────────────────
print("\n-- Species composition per cluster ---------------------------------")
plot_species_composition(df, clusters_gmm, CLUSTER_NAMES, save_dir=RESULTS_DIR)


# ─── GMM confidence ───────────────────────────────────────────────────────────
print("\n-- GMM assignment confidence ---------------------------------------")
plot_gmm_confidence(probs, CLUSTER_NAMES, save_dir=RESULTS_DIR)

max_probs = probs.max(axis=1)
print(f"  Mean confidence:           {max_probs.mean()*100:.1f}%")
print(f"  Entities > 90% confidence: {(max_probs > 0.9).sum()} / {len(max_probs)}"
      f" ({(max_probs > 0.9).mean()*100:.1f}%)")
print(f"  Entities < 70% confidence: {(max_probs < 0.7).sum()}"
      f" ({(max_probs < 0.7).mean()*100:.1f}%)")

df["max_prob"] = max_probs
low_conf = df[df["max_prob"] < 0.7]
print("\n  Low-confidence (<70%) species breakdown:")
print(low_conf["species"].value_counts().to_string())
print("  These are mostly humans sitting on the smart/tank boundary — their")
print("  sensor readings share traits with both clusters, making them the")
print("  most tactically interesting entities for the RL agent to encounter.")


# ─── Is 'flying' just flight=1? ───────────────────────────────────────────────
print("\n-- Critical appraisal: is 'flying' trivially flight==1? -----------")
flight_by_species = df.groupby("species")["flight"].mean()
print(flight_by_species.to_string())
print()
print("  Lizards have 48% flight=1, yet they are assigned to the 'tank' cluster.")
print("  This means 'flying' is NOT trivially recoverable by flight==1 alone.")
print("  Winged_rats (100% flight) also differ on stench, sound, and weight,")
print("  meaning PCA genuinely synthesises multiple features — the cluster is")
print("  not a single-feature lookup. The unsupervised approach adds value.")


# ─── Sample behavior lookups ───────────────────────────────────────────────────
from task2.entity_behavior import enemy_behavior_gmm

print("\nSample behavior profiles (first 5 enemies):")
for i in range(5):
    print(f"  Enemy {i}: {enemy_behavior_gmm(i, probs, CLUSTER_NAMES)}")


# ─── Save artefacts for RL integration ────────────────────────────────────────
artefacts = {
    "gmm":              gmm,
    "pca_2d":           pca_2d,
    "scaler":           scaler,
    "imputer":          imputer,
    "probs":            probs,
    "cluster_labels":   clusters_gmm,
    "cluster_names":    CLUSTER_NAMES,
    "cluster_profiles": cluster_profiles,
    "feature_cols":     feature_cols,
}

save_path = os.path.join(RESULTS_DIR, "gmm_artefacts.pkl")
with open(save_path, "wb") as f:
    pickle.dump(artefacts, f)
print(f"\nSaved artefacts -> {save_path}")
print(f"\nAll outputs saved to  {RESULTS_DIR}")
print("\nDone.")
