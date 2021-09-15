from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from scipy.spatial.distance import cdist, pdist
import numpy as np
from metrics import getSilhouette

# WARNING cannot use matplotlib in colab from here ...
# import matplotlib.pyplot as plt

def ComputeWSSBSS(X, pdist_matrix_sqr, Z, metric, k_values=range(1,20), postprocessing=False):
    wss_values = []
    bss_values = []

    for k in k_values:
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        _, _, cluster_labels = getSilhouette(distance_matrix, cluster_labels, postprocessing)
        frequency = np.bincount(cluster_labels)
        index = np.nonzero(frequency)[0]
        
        centroids = [np.mean(X[cluster_labels==c],axis=0) for c in index]
        
        D = cdist(X, centroids, metric)
        cIdx = np.argmin(D,axis=1)
        d = np.min(D,axis=1)

        avgWithinSS = sum(d)/len(X)

        # Total with-in sum of square
        wss = sum(d**2)

        tss = np.sum(pdist_matrix_sqr)/len(X)

        bss = tss-wss

        wss_values += [wss]
        bss_values += [bss]
    return wss_values,bss_values

def PlotKneeElbow(bss_values,wss_values,k_values,title="", save_path=None):
    fig = plt.figure(figsize=(24,12))
    font = {'family' : 'sans', 'size'   : 12}
    plt.rc('font', **font)
    plt.plot(k_values, wss_values, 'bo-', color='red', label='WSS')
    plt.plot(k_values, bss_values, 'bo-', color='blue', label='BSS')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('BSS & WSS')
    plt.xticks(k_values)
    plt.legend()
    plt.title(title);

    plt.savefig(save_path)