"""k selection, K-Means, Agglomerative."""
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


def pick_k(Z, ks=range(4, 16), plot_path=None, seed=42):
    scores = {}
    for k in ks:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
        scores[k] = silhouette_score(Z, labels, sample_size=3000, random_state=seed)
    if plot_path:
        plt.figure(figsize=(6, 3.5))
        plt.plot(list(scores), list(scores.values()), marker="o")
        plt.xlabel("k"); plt.ylabel("Silhouette"); plt.title("Choosing k")
        plt.tight_layout(); plt.savefig(plot_path, dpi=150); plt.close()
    return max(scores, key=scores.get), scores


def fit_both(Z, k, seed=42):
    km = KMeans(k, n_init=20, random_state=seed).fit(Z)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    return km.labels_, agg


def compare(Z, km_labels, agg_labels, seed=42):
    return {
        "kmeans_silhouette": silhouette_score(Z, km_labels, sample_size=3000, random_state=seed),
        "agglo_silhouette": silhouette_score(Z, agg_labels, sample_size=3000, random_state=seed),
        "adjusted_rand_index": adjusted_rand_score(km_labels, agg_labels),
    }


def plot_2d(Z, labels, title, path, seed=42):
    xy = PCA(n_components=2, random_state=seed).fit_transform(Z)
    plt.figure(figsize=(8, 6))
    plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20", s=6, alpha=0.7)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()