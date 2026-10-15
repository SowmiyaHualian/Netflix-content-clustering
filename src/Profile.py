"""Top terms and cluster summaries."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def top_terms(tfidf, X, labels, n=10):
    terms = np.array(tfidf.get_feature_names_out())
    out = {}
    for c in sorted(set(labels)):
        centroid = np.asarray(X[labels == c].mean(axis=0)).ravel()
        out[c] = terms[centroid.argsort()[::-1][:n]].tolist()
    return out


def profile_clusters(df, label_col, terms):
    rows = []
    for c, g in df.groupby(label_col):
        rows.append({
            "cluster": c,
            "size": len(g),
            "pct_tv": round((g["type"] == "TV Show").mean() * 100, 1),
            "median_year": int(g["release_year"].median()),
            "top_genres": ", ".join(g["listed_in"].str.split(", ").explode().value_counts().head(3).index),
            "top_terms": ", ".join(terms[c]),
            "examples": " | ".join(g["title"].sample(min(3, len(g)), random_state=1)),
        })
    return pd.DataFrame(rows)


def plot_sizes(df, label_col, path):
    counts = df[label_col].value_counts().sort_index()
    plt.figure(figsize=(7, 3.5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Cluster"); plt.ylabel("Titles"); plt.title("Cluster sizes")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()