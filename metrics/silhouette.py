from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils import check_X_y
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.cluster.hierarchy import fcluster

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import functools

def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.
    Parameters
    ----------
    n_labels : int
        Number of labels.
    n_samples : int
        Number of samples.
    """
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)
        
def _silhouette_reduce(D_chunk, start, labels, label_freqs, small_clusters_mask):
    """Accumulate silhouette statistics for vertical chunk of X.
    Parameters
    ----------
    D_chunk : array-like of shape (n_chunk_samples, n_samples)
        Precomputed distances for a chunk.
    start : int
        First index in the chunk.
    labels : array-like of shape (n_samples,)
        Corresponding cluster labels, encoded as {0, ..., n_clusters-1}.
    label_freqs : array-like
        Distribution of cluster labels in ``labels``.
    """
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)),
                           dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(labels, weights=D_chunk[i],
                                      minlength=len(label_freqs))

    #print(clust_dists)
    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start:start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    # ignore small clusters
    clust_dists[:,small_clusters_mask] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists, clust_dists

def silhouette_samples(X, labels, *, metric='euclidean', cluster_size_threshold=200, **kwds):
    X, labels = check_X_y(X, labels, accept_sparse=['csc', 'csr'])

    # Check for non-zero diagonal entries in precomputed distance matrix
    if metric == 'precomputed':
        atol = np.finfo(X.dtype).eps * 100
        if np.any(np.abs(np.diagonal(X)) > atol):
            raise ValueError(
                'The precomputed distance matrix contains non-zero '
                'elements on the diagonal. Use np.fill_diagonal(X, 0).'
            )

    le = LabelEncoder()
    unique_labels = np.unique(labels)
    has_outliers = unique_labels[0] == -1
    labels = le.fit_transform(labels)
    n_samples = len(labels)
    label_freqs = np.bincount(labels)
    check_number_of_labels(len(le.classes_), n_samples)
    small_clusters_mask = label_freqs < cluster_size_threshold
    # this would be the cluster -1 (outliers)
    if has_outliers:
        # dont't consider it
        # print('Has outliers')
        small_clusters_mask[0] = True
    #if cluster_size_threshold:
     # print(f'Ignoring clusters of size less than {cluster_size_threshold}:')
     # print(unique_labels[small_clusters_mask])

    kwds['metric'] = metric
    reduce_func = functools.partial(_silhouette_reduce,
                                    labels=labels, label_freqs=label_freqs, small_clusters_mask=small_clusters_mask)
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func,
                                              **kwds))
    intra_clust_dists, inter_clust_dists, clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)
    clust_dists = np.concatenate(clust_dists)

    #print("intra_clust_dists: ", intra_clust_dists.shape)
    #print("inter_clust_dists: ", inter_clust_dists.shape)
    #print("clust_dists: ", clust_dists.shape)

    denom = (label_freqs - 1).take(labels, mode='clip')
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    sil_samples = inter_clust_dists - intra_clust_dists
    #mask = sil_samples<0.0
    #index = 5
    #print(index, sil_samples[index], clust_dists[3], intra_clust_dists[index])
    with np.errstate(divide="ignore", invalid="ignore"):
        sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples), clust_dists, small_clusters_mask

def fixSilhouette(distance_matrix, cluster_labels, sample_silhouette_values, clust_dists, metric="precomputed"):
    mask = sample_silhouette_values<0.0
    cluster_labels[mask] = np.unique(cluster_labels)[np.argmin(clust_dists[mask], axis=1)]
    # print(sample_silhouette_values[mask])

    return silhouette_samples(distance_matrix, cluster_labels, metric=metric)

def setOutliers(cluster_labels, sample_silhouette_values, cluster_size_threshold):
    clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    small_clusters_mask = cluster_counts < cluster_size_threshold
    # print(clusters, cluster_counts, small_clusters_mask)
    small_clusters = clusters[small_clusters_mask]

    mask = np.array(sample_silhouette_values<0.0) | np.isin(cluster_labels, small_clusters)
    cluster_labels[mask] = -1

    return cluster_labels

def removeOutliersFromAvg(silhouette_avg, sil_values):
    ## not used ##
    # returns the adjusted average without outliers
    return (silhouette_avg * len(sil_values) - np.sum(sil_values[sil_values<0.0]))/(len(sil_values)-len(sil_values[sil_values<0.0]))*1.0


def getSilhouetteAvg(sample_silhouette_values, cluster_labels, small_clusters_mask):
    small_clusters = np.unique(cluster_labels)[small_clusters_mask]
    mask = ~np.isin(cluster_labels, small_clusters)
    return np.average(sample_silhouette_values[mask])

def getSilhouette(distance_matrix, cluster_labels, postprocessing=False, cluster_size_threshold=200):

    # Compute the silhouette scores for each sample
    sample_silhouette_values, clust_dists, _ = silhouette_samples(distance_matrix, cluster_labels, metric="precomputed")

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = np.average(sample_silhouette_values)
    print("For n_clusters =", len(np.unique(cluster_labels)),
          "The average silhouette_score is :", silhouette_avg)

    if postprocessing == True:
        prev_fix_score = silhouette_avg
        prev_cluster_labels = np.copy(cluster_labels)
        prev_sample_silhouette_values = np.copy(sample_silhouette_values)
        # prev_indices = np.where(sample_silhouette_values<0.0)[0]
        while np.any(sample_silhouette_values<0.0):
            sample_silhouette_values, _, small_clusters_mask = fixSilhouette(distance_matrix, 
                cluster_labels, sample_silhouette_values, clust_dists, metric="precomputed")
            # print("Prev:", np.average(sample_silhouette_values[sample_silhouette_values >= 0]))
            silhouette_avg = getSilhouetteAvg(sample_silhouette_values, cluster_labels, small_clusters_mask)

            indices = np.where(sample_silhouette_values<0.0)[0]
            # prev_indices = np.intersect1d(prev_indices, indices)
            # print("Common indices:", prev_indices)
            # print("First still to fix: ", indices[0])
            print("To be fixed len: ", indices.shape)
            print("While fix: For n_clusters =", len(np.unique(cluster_labels)),
              "The average silhouette_score is :", silhouette_avg)
            # print(np.unique(cluster_labels))

            if silhouette_avg - prev_fix_score < 0.01*prev_fix_score:
                if silhouette_avg < prev_fix_score:
                    print("Score lowered, keeping previous iteration labels..")
                    cluster_labels = prev_cluster_labels
                    sample_silhouette_values = prev_sample_silhouette_values
                break
            else:
                prev_fix_score = silhouette_avg
                prev_cluster_labels = np.copy(cluster_labels)
                prev_sample_silhouette_values = np.copy(sample_silhouette_values)
        
        # Set outliers
        cluster_labels = setOutliers(cluster_labels, sample_silhouette_values, cluster_size_threshold)
        # Compute the silhouette scores for each sample
        sample_silhouette_values, _, small_clusters_mask = silhouette_samples(distance_matrix, cluster_labels, metric="precomputed")
        silhouette_avg = getSilhouetteAvg(sample_silhouette_values, cluster_labels, small_clusters_mask)
        print(">>> After fix: For n_clusters =", len(np.unique(cluster_labels)), "(including outliers).",
          "The average silhouette_score is :", silhouette_avg)
      
    return silhouette_avg, sample_silhouette_values, cluster_labels
