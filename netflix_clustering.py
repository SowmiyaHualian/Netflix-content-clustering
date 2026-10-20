"""
Netflix / OTT Shows Clustering & Segmentation

Pipeline: clean -> TF-IDF (description + genres) + metadata -> TruncatedSVD
          -> K-Means vs Agglomerative -> cluster profiling -> saved outputs.

Usage:
    python netflix_clustering.py
    python netflix_clustering.py --data data/netflix_titles.csv --k 8 --components 100
"""
import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # save figures to disk, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, Normalizer, OneHotEncoder

warnings.filterwarnings("ignore")
SEED = 42
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def split_genres(s):
    return [g.strip() for g in s.split(",")]


def load_and_clean(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(df["type"].value_counts().to_string(), "\n")

    df = df.drop_duplicates(subset=["title", "type", "release_year"]).reset_index(drop=True)
    df["description"] = df["description"].fillna("")
    df["listed_in"] = df["listed_in"].fillna("Unknown")

    # Some rows in the Kaggle file have a duration (e.g. "74 min") in the rating column
    df["rating"] = df["rating"].fillna("Unknown")
    df.loc[df["rating"].astype(str).str.contains("min", case=False), "rating"] = "Unknown"

    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df["release_year"] = df["release_year"].fillna(df["release_year"].median())
    return df


def build_features(df):
    pre = ColumnTransformer(
        [
            ("desc", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3,
                                     max_df=0.5, sublinear_tf=True), "description"),
            ("genre", TfidfVectorizer(tokenizer=split_genres, token_pattern=None,
                                      lowercase=False), "listed_in"),
            ("meta", OneHotEncoder(handle_unknown="ignore"), ["type", "rating"]),
            ("year", MinMaxScaler(), ["release_year"]),
        ],
        transformer_weights={"desc": 1.0, "genre": 1.0, "meta": 0.4, "year": 0.8},
    )
    X = sparse.csr_matrix(pre.fit_transform(df))
    print(f"Feature matrix: {X.shape}")
    return X, pre.get_feature_names_out()


def reduce_dims(X, n_components, fig_dir):
    # TruncatedSVD (LSA) is the sparse-friendly counterpart of PCA
    n_components = min(n_components, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    Z = Normalizer().fit_transform(svd.fit_transform(X))
    cum = np.cumsum(svd.explained_variance_ratio_)
    print(f"{n_components} SVD components keep {cum[-1]:.1%} of the variance")

    plt.figure()
    plt.plot(range(1, n_components + 1), cum)
    plt.xlabel("SVD components")
    plt.ylabel("Cumulative explained variance")
    plt.title("TruncatedSVD explained variance")
    plt.savefig(f"{fig_dir}/01_svd_variance.png", bbox_inches="tight")
    plt.close()
    return Z


def choose_k(Z, fig_dir, k_min=2, k_max=15):
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(5000, len(Z)), random_state=SEED)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    scores = pd.DataFrame(rows).set_index("k")
    print("\nk selection:\n", scores.round(3).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    scores["inertia"].plot(marker="o", ax=ax[0], title="Elbow (inertia)")
    scores["silhouette"].plot(marker="o", ax=ax[1], title="Silhouette")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/02_choose_k.png", bbox_inches="tight")
    plt.close()

    # Skip trivial k < 5 (a 2-cluster split is usually just Movie vs TV Show)
    return int(scores.loc[5:, "silhouette"].idxmax())


def evaluate(Z, labels):
    return {
        "silhouette": silhouette_score(Z, labels, sample_size=min(5000, len(Z)), random_state=SEED),
        "davies_bouldin (lower=better)": davies_bouldin_score(Z, labels),
        "calinski_harabasz": calinski_harabasz_score(Z, labels),
    }


def plot_2d(Z, df, k, fig_dir):
    xy = PCA(n_components=2, random_state=SEED).fit_transform(Z)  # for plotting only
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for a, col, title in zip(ax, ["cluster_kmeans", "cluster_agglo"],
                             ["K-Means", "Agglomerative (ward)"]):
        a.scatter(xy[:, 0], xy[:, 1], c=df[col], cmap="tab10", s=5, alpha=0.6)
        a.set_title(f"{title}, k={k}")
        a.set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/03_clusters_2d.png", bbox_inches="tight")
    plt.close()


def profile_clusters(df, X, feature_names, fig_dir):
    text_cols = [i for i, n in enumerate(feature_names) if n.startswith(("desc__", "genre__"))]
    text_names = np.array([feature_names[i].split("__", 1)[1] for i in text_cols])
    Xt = X[:, text_cols]

    def top_terms(c, n=8):
        mean = np.asarray(Xt[(df["cluster"] == c).values].mean(axis=0)).ravel()
        return ", ".join(text_names[np.argsort(mean)[::-1][:n]])

    def top_genres(s, n=3):
        return ", ".join(s.str.split(", ").explode().value_counts().head(n).index)

    g = df.groupby("cluster")
    profile = pd.DataFrame({
        "size": g.size(),
        "share_%": (g.size() / len(df) * 100).round(1),
        "tv_show_%": (g["type"].apply(lambda s: (s == "TV Show").mean()) * 100).round(1),
        "median_year": g["release_year"].median(),
        "top_genres": g["listed_in"].apply(top_genres),
        "top_terms": [top_terms(c) for c in sorted(df["cluster"].unique())],
        "examples": g["title"].apply(
            lambda s: " | ".join(s.sample(min(3, len(s)), random_state=SEED))),
    })

    # Genre x cluster heatmap
    gen = (df[["cluster", "listed_in"]]
           .assign(genre=df["listed_in"].str.split(", "))
           .explode("genre").reset_index(drop=True))
    top15 = gen["genre"].value_counts().head(15).index
    heat = pd.crosstab(gen["cluster"], gen["genre"], normalize="index")[top15] * 100
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(heat) + 2))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(top15)))
    ax.set_xticklabels(top15, rotation=60, ha="right")
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels([f"Cluster {c}" for c in heat.index])
    plt.colorbar(im, label="% of cluster")
    ax.set_title("Genre mix by cluster")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/04_genre_heatmap.png", bbox_inches="tight")
    plt.close()

    # Release year by cluster
    fig, ax = plt.subplots(figsize=(9, 4))
    df.boxplot(column="release_year", by="cluster", ax=ax, grid=False)
    ax.set_title("Release year by cluster")
    plt.suptitle("")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Release year")
    plt.savefig(f"{fig_dir}/05_year_by_cluster.png", bbox_inches="tight")
    plt.close()
    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("NETFLIX_CSV", "data/netflix_titles.csv"))
    ap.add_argument("--out", default="data", help="folder for CSV outputs")
    ap.add_argument("--figs", default="figures", help="folder for figures")
    ap.add_argument("--k", type=int, default=None, help="number of clusters (default: auto)")
    ap.add_argument("--components", type=int, default=100, help="SVD components")
    ap.add_argument("--similar", default=None, help="print titles similar to this one")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figs, exist_ok=True)

    df = load_and_clean(args.data)
    X, feature_names = build_features(df)
    Z = reduce_dims(X, args.components, args.figs)

    k = args.k or choose_k(Z, args.figs)
    print(f"\nUsing k = {k}")

    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
    ag = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
    df["cluster_kmeans"] = km.labels_
    df["cluster_agglo"] = ag.labels_

    metrics = pd.DataFrame({"KMeans": evaluate(Z, km.labels_),
                            "Agglomerative (ward)": evaluate(Z, ag.labels_)}).T
    print("\nModel comparison:\n", metrics.round(3).to_string())
    print("Adjusted Rand index between methods:",
          round(adjusted_rand_score(km.labels_, ag.labels_), 3))
    print("Sizes KMeans:", np.bincount(km.labels_).tolist())
    print("Sizes Agglo: ", np.bincount(ag.labels_).tolist())

    best = ("cluster_kmeans"
            if metrics.loc["KMeans", "silhouette"] >= metrics.loc["Agglomerative (ward)", "silhouette"]
            else "cluster_agglo")
    df["cluster"] = df[best]
    print("Primary clustering:", best)

    plot_2d(Z, df, k, args.figs)
    profile = profile_clusters(df, X, feature_names, args.figs)
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        print("\nCluster profiles:\n", profile.to_string())

    df.drop(columns=["cluster_kmeans", "cluster_agglo"]).to_csv(f"{args.out}/netflix_clustered.csv", index=False)
    profile.to_csv(f"{args.out}/cluster_profiles.csv")
    metrics.to_csv(f"{args.out}/model_metrics.csv")
    print(f"\nSaved CSVs to {args.out}/ and figures to {args.figs}/")

    if args.similar:
        idx = df.index[df["title"].str.lower() == args.similar.lower()]
        if len(idx) == 0:
            print(f"'{args.similar}' not found")
        else:
            nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(Z)
            _, nbrs = nn.kneighbors(Z[idx[0]].reshape(1, -1))
            print(f"\nTitles similar to {args.similar}:")
            print(df.loc[nbrs[0][1:], ["title", "type", "release_year", "listed_in", "cluster"]].to_string())


if __name__ == "__main__":
    main()