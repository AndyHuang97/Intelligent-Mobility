import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from scipy.spatial.distance import cdist, pdist

# Assumes that there is only one split at each cut k
def get_split_cluster_labels(Z, k):
    cluster_init_label = 1
    split_cluster_labels = []

    k1 = k - 1
    k2 = k

    cluster_labels_k1 = fcluster(Z, k1, criterion='maxclust')
    cluster_labels_k2 = fcluster(Z, k2, criterion='maxclust')

    k1_counts = np.unique(cluster_labels_k1, return_counts=True)
    k2_counts = np.unique(cluster_labels_k2, return_counts=True)
    split_sizes = []
    k2_splits = []

    j = 0
    sum = 0
    for i in range(len(k2_counts[0])):
        if k1_counts[1][i-j] != k2_counts[1][i]:
            # print(k1_counts[0][i-j], k1_counts[1][i-j], k2_counts[0][i], k2_counts[1][i], sep='\t')
            split_sizes.append(k1_counts[1][i-j])
            k2_splits.append((k2_counts[0][i], k2_counts[1][i]))
            # there has been a split, so k2 has one more label (the one that came from the k1 one)
            sum += k2_counts[1][i]
            if sum < k1_counts[1][i-j]:
                j += 1
            elif sum == k1_counts[1][i-j]:
                # initialize for next split
                sum = 0
            else:
                raise ValueError("split not correct")
    split_sizes = np.array(split_sizes)
    k2_splits = np.array(k2_splits)
    max_split = np.max(split_sizes)
    k_mask = np.flatnonzero(split_sizes == max_split)
    k2_selected = k2_splits[k_mask]
    print(f'max_split: {max_split}')
    print(k2_selected)
    split_cluster_labels = k2_selected[k2_selected[:,1] > split_th,0]
    print(split_cluster_labels)
    return split_cluster_labels

def reassign_labels(cluster_labels_k1, cluster_labels_k2):
    k1_values = np.unique(cluster_labels_k1)
    k2_values = np.unique(cluster_labels_k2)

    first_unassigned_value = k1_values[-1] + 1
    
    cluster_labels_k2 += first_unassigned_value
    
    return cluster_labels_k2

def merge_multiple_cuts(Z, k_values, split_th=200):
    k1 = k_values[0]
    cluster_labels_k1 = fcluster(Z, k1, criterion='maxclust')
    for k2 in k_values[1:]:
        split_cluster_labels = get_split_cluster_labels(Z, k=k2, split_th=split_th)

        cluster_labels_k2 = fcluster(Z, k2, criterion='maxclust')
        print(f"k1({k1}):", np.unique(cluster_labels_k1))
        print(f"k2({k2}):", split_cluster_labels)
        split_cluster_mask = np.isin(cluster_labels_k2, split_cluster_labels)
        print(np.unique(cluster_labels_k1[split_cluster_mask]), np.unique(cluster_labels_k2[split_cluster_mask]))
        cluster_labels_k2 = reassign_labels(cluster_labels_k1, cluster_labels_k2)
        print(np.unique(cluster_labels_k1[split_cluster_mask]), np.unique(cluster_labels_k2[split_cluster_mask]))
        cluster_labels_k1[split_cluster_mask] = cluster_labels_k2[split_cluster_mask]
    return cluster_labels_k1
