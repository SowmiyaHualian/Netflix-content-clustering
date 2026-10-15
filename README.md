# Netflix Content Clustering

Grouping Netflix movies and TV shows into clusters based on descriptions,
genres and release era, to uncover hidden content patterns.

## Approach
1. Text feature = description + genres + type + release-era token
2. TF-IDF (unigrams + bigrams)
3. TruncatedSVD (100 components) + L2 normalisation
4. k chosen via silhouette score
5. K-Means vs Agglomerative (Ward) comparison

## Results
| Cluster | Name | Size | Top genres | Example titles |
|---|---|---|---|---|
| 0 | <your label> | ... | ... | ... |

![Clusters](images/clusters_kmeans.png)

## Limitations
Silhouette scores are modest on text data and some clusters overlap...

## Run it
pip install -r requirements.txt
Dataset: https://www.kaggle.com/datasets/shivamb/netflix-shows

## Next steps
Sentence-transformer embeddings, similar-show recommender
