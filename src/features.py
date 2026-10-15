"""TF-IDF + dimensionality reduction."""
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer


def tfidf_features(texts, max_features=8000):
    tfidf = TfidfVectorizer(
        stop_words="english", max_features=max_features,
        ngram_range=(1, 2), min_df=3, sublinear_tf=True,
    )
    return tfidf, tfidf.fit_transform(texts)


def reduce(X, n_components=100, seed=42):
    """TruncatedSVD (PCA for sparse data) + L2 normalisation."""
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    Z = Normalizer(copy=False).fit_transform(svd.fit_transform(X))
    return svd, Z