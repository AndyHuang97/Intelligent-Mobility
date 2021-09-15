import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from scipy.cluster.hierarchy import fcluster
from metrics.silhouette import getSilhouette
from postprocessing.merge_cuts import merge_multiple_cuts

def plotSilhouette(df, fig, silhouette_avg, sample_silhouette_values, cluster_labels, silhouette_row=1, silhouette_col=1):

    x_lower = 10

    for i in np.unique(cluster_labels):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        x_upper = x_lower + size_cluster_i

        #colors = plt.cm.Spectral(cluster_labels.astype(float) / n_clusters)

        filled_area = go.Scatter(x=np.arange(x_lower, x_upper),
                                y=ith_cluster_silhouette_values,
                                mode='lines',
                                name=str(i),
                                #showlegend=True,
                                line=dict(width=0.5,
                                        #color=colors
                                        ),
                                fill='tozeroy')
        fig.add_trace(filled_area, silhouette_row, silhouette_col)
      
        # Compute the new y_lower for next plot
        x_lower = x_upper + 10  # 10 for the 0 samples

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 
    fig.update_yaxes(title_text='The silhouette coefficient values',
                     row=silhouette_row, col=silhouette_col,
                     range=[-1, 1])

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    fig.update_xaxes(title_text='Cluster label',
                     row=silhouette_row, col=silhouette_col,
                     #showticklabels=False,
                     range=[0, len(df) + (len(np.unique(cluster_labels)) + 1) * 10])

    # The vertical line for average silhouette score of all the values
    axis_line = go.Scatter(y=[silhouette_avg]*100,
                           x=np.linspace(0, len(df), 100,),
                           #showlegend=True,
                           name='silhouette avg',
                           mode='lines',
                           line=dict(color="red", dash='dash',
                                     width =1) )

    fig.add_trace(axis_line, silhouette_row, silhouette_col)

    return fig

def plot_silhouette_from_k_values(df, distance_matrix, Z, metric, link, k_values, postprocessing, row_height=300, save_path=None, showFig=False):
    # Customization options
    fontsize = 8

    #
    titles = []
    for k in k_values:
        titles.append(f"Silhouette for k={k} clusters with {metric}{vars} and {link}")

    rows = len(k_values)
    fig = make_subplots(rows=rows, cols=1,
                        vertical_spacing=0.4/rows,
                        subplot_titles=titles,
                        )
    
    for i in range(rows):
        k = k_values[i]

        cluster_labels = fcluster(Z, k, criterion='maxclust')

        silhouette_avg, sample_silhouette_values, cluster_labels = getSilhouette(distance_matrix, cluster_labels, postprocessing)
        titles[i] = f"{titles[i]}, avg sil:{silhouette_avg:.02f}"
        k = np.unique(cluster_labels).shape[0]
        # add silhouette row
        fig = plotSilhouette(df,
                            fig, 
                            k, 
                            silhouette_avg, sample_silhouette_values, cluster_labels,
                            silhouette_row=i+1) # plotly starts from 1!
        fig.layout.annotations[i].update(text=titles[i])

    # Update layout
    fig.update_layout(
        #title=f"silhouette_avg: {silhouette_avg}",
        height=row_height*rows,
        showlegend=False,
        bargap=0.05,
        font=dict(size=fontsize)
    )


    if save_path:
        fig.write_html(save_path)
    if showFig:
        fig.show()

def plot_silhouette_merge_k_values(df, distance_matrix, Z, metric, link, k_values, postprocessing, row_height=300, save_path=None, showFig=False):
    cut_values = []
    rows = len(k_values)
    title_placeholders = ["title"]*rows
    fig = make_subplots(rows=rows, cols=1, subplot_titles=title_placeholders)

    for i, k in enumerate(k_values):
        cut_values.append(k)
        cluster_labels = merge_multiple_cuts(Z, cut_values)
        silhouette_avg, sample_silhouette_values, cluster_labels = getSilhouette(distance_matrix, cluster_labels, postprocessing)

        fig = plotSilhouette(df,
                                fig=fig,
                                silhouette_avg=silhouette_avg,
                                sample_silhouette_values=sample_silhouette_values,
                                cluster_labels=cluster_labels,
                                silhouette_row=i+1)

        title = f"Cuts with k={cut_values} clusters, {metric}{df.shape[1]} and link {link}, with silhouette average: {silhouette_avg}"
        fig.layout.annotations[i].update(text=title)
    fig.update_layout(height=row_height*rows,
                      showlegend=False)

    if save_path:
        fig.write_html(save_path)
    if showFig:
        fig.show()
  