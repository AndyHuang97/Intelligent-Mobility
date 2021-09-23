from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
import pandas as pd
from .silhouette import getSilhouette

# WARNING cannot use matplotlib in colab from here ...
# import matplotlib.pyplot as plt

# Compute wss and bss with centroid
def ComputeWSSBSS(X, Z, distance_matrix, metric, k_values=range(1,20), postprocessing=False):
    wss_values = []
    bss_values = []

    prev_cluster_labels = np.array([-1])
    for k in k_values:
        print("k:",k)
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        cluster_labels = np.array(cluster_labels)
        
        if (prev_cluster_labels == cluster_labels).all():
            wss_values += [wss]
            bss_values += [bss]
            continue
        else:
            prev_cluster_labels = cluster_labels.copy()
        _, _, cluster_labels = getSilhouette(distance_matrix, cluster_labels, postprocessing)
        #outlier_label = 0
        #cluster_labels = np.where(cluster_labels==-1, outlier_label, cluster_labels)
        clusters, frequency = np.unique(cluster_labels, return_counts=True)
        no_outliers = clusters > 0
        
        centroids = [np.mean(X[cluster_labels==c],axis=0) for c in clusters[no_outliers]]
        
        D = cdist(X, centroids, metric)
        cIdx = np.argmin(D,axis=1)
        d = np.min(D,axis=1)

        avgWithinSS = np.sum(d)/len(X)

        # Total with-in sum of square
        wss = np.sum(d**2)

        tss = np.sum(distance_matrix**2)/len(X)

        bss = tss-wss

        wss_values += [wss]
        bss_values += [bss]
    return wss_values,bss_values

# Get the medoid for wss computation.
# df_cluster is expected to be converted in an appropriate form.
def getMedoid(df_cluster, metric):
    distance_matrix_cluster = squareform(pdist(df_cluster, metric=metric))
    distance_matrix_medoid_ix = np.argmin(distance_matrix_cluster.sum(axis=0))
    medoid = df_cluster.iloc[distance_matrix_medoid_ix]

    return medoid

# Get the medoid using the vdm metric for a radar chart.
# vdm_df_cluster is expected to be converted in an appropriate form.
# df_cluster is the original dataframe, which is the one we want to take information from.
# This variant is useful when we want to represent the original value of the medoid in a radar chart.
def getMedoidVDM(vdm_df_cluster, df_cluster, metric):
    vdm_df_cluster_reduced = vdm_df_cluster.copy()
    df_cluster_reduced = df_cluster.copy()
    vdm_df_cluster_reduced.reset_index(drop=True, inplace=True)
    df_cluster_reduced.reset_index(drop=True, inplace=True)

    distance_matrix_cluster = squareform(pdist(vdm_df_cluster_reduced, metric=metric))
    distance_matrix_medoid_ix = np.argmin(distance_matrix_cluster.sum(axis=0))
    medoid = df_cluster_reduced.iloc[distance_matrix_medoid_ix]

    return medoid

# Get the ith medoid using the vdm metric for a radar chart.
# vdm_df_cluster is expected to be converted in an appropriate form.
# df_cluster is the original dataframe, which is the one we want to take information from.
# This variant is useful when we want to represent the original value of the medoid in a radar chart.
def get_ith_medoid(vdm_df_cluster, df_cluster, metric, ith):
    vdm_df_cluster_reduced = vdm_df_cluster.copy()
    df_cluster_reduced = df_cluster.copy()
    vdm_df_cluster_reduced.reset_index(drop=True, inplace=True)
    df_cluster_reduced.reset_index(drop=True, inplace=True)

    distance_matrix_cluster = squareform(pdist(vdm_df_cluster_reduced, metric=metric))
    sums = distance_matrix_cluster.sum(axis=0)
    sorted_sums = np.sort(sums)
    print(sorted_sums)
    ith_medoid = sorted_sums[ith]
    medoid_ix = np.where(sums == ith_medoid)[0][0]
    medoid = df_cluster_reduced.iloc[medoid_ix]
    return medoid

# Compute the wss using the medoid.
def ComputeWSS(df, Z, metric, k_values=range(1,20), postprocessing=False, save_path=None):
    wss_values = []

    prev_cluster_labels = np.array([-1])
    for k in k_values:
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        cluster_labels = np.array(cluster_labels)

        if (prev_cluster_labels == cluster_labels).all():
            wss_values += [wss]
            if save_path:
                np.save(save_path, wss_values)
            continue
        else:
            prev_cluster_labels = cluster_labels.copy()
        #distance_matrix = squareform(pdist(df, metric=metric))
        #_, _, cluster_labels = getSilhouette(distance_matrix, cluster_labels, postprocessing)
        #del distance_matrix
        df["cluster"] = cluster_labels
    
        df = df.copy().loc[df["cluster"] != "-1"] # Filter outliers
        cluster_values = np.unique(cluster_labels)
        cluster_values = cluster_values[cluster_values > 0] # do not get outlier label

        medoids = []
        for cluster in cluster_values:
            df_cluster = df.loc[df["cluster"]==cluster].drop("cluster", axis=1)
            medoid = getMedoid(df_cluster, metric)
            medoids.append(medoid)
        medoids_df = pd.DataFrame(medoids)
        #print("medoids_df: ", medoids_df)
        df = df.drop("cluster", axis=1)

        D = cdist(df, medoids_df, metric)
        #print("D: ", D)
        cIdx = np.argmin(D,axis=1)
        #print("cIdx: ", cIdx)
        d = np.min(D,axis=1)
        #print("d: ", d)

        #avgWithinSS = np.sum(d)/df.shape[0]
        #print("avgWithinSS: ", avgWithinSS)
        # Total with-in sum of square
        wss = np.sum(d**2)
        #print("wss: ", wss)
        wss_values += [wss]

        if save_path:
            np.save(save_path, wss_values)

    return wss_values