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
  #print(k1_counts)
  #print(k2_counts)
  for i in range(len(k1_counts[0])):
    if k1_counts[1][i] != k2_counts[1][i]:
      break

  split_cluster_labels = [i+cluster_init_label,i+1+cluster_init_label]
  return split_cluster_labels

def reassign_labels(cluster_labels_k1, cluster_labels_k2):
  k1_values = np.unique(cluster_labels_k1)
  k2_values = np.unique(cluster_labels_k2)

  first_unassigned_value = k1_values[-1] + 1
  
  cluster_labels_k2 += first_unassigned_value
  
  return cluster_labels_k2

def merge_multiple_cuts(Z, k_values,):
  k1 = k_values[0]
  cluster_labels_k1 = fcluster(Z, k1, criterion='maxclust')
  for k2 in k_values[1:]:
    split_cluster_labels = get_split_cluster_labels(Z, k=k2)

    cluster_labels_k2 = fcluster(Z, k2, criterion='maxclust')
    print("k1:", np.unique(cluster_labels_k1))
    print("k2:", split_cluster_labels)
    split_cluster_mask = np.isin(cluster_labels_k2, split_cluster_labels)
    cluster_labels_k2 = reassign_labels(cluster_labels_k1, cluster_labels_k2)

    cluster_labels_k1[split_cluster_mask] = cluster_labels_k2[split_cluster_mask]
  return cluster_labels_k1