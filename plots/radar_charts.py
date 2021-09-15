import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import numpy as np

def plot_individual_radar(fig, polar_args, r, theta, index, row=1, col=1):
  fig.add_trace(go.Scatterpolar(
                        r=r,
                        theta=theta,
                        showlegend=False,
                        fill='toself',
                        line_color=px.colors.qualitative.Plotly[index%len(px.colors.qualitative.Plotly)]), 
                    row=row, col=col
                  )
  polar_args[f"polar{index}"] = dict(radialaxis=dict(visible=True,
                                                      range=[0.0, 1.0]
                                                      ),
                                     )

  return fig, polar_args

def plot_all_medoids_radar(medoids_df, cluster_labels_counts, plot_cols=3,row_height=300, save_path=None):
  cols = plot_cols
  rows = int(np.ceil(medoids_df.shape[0]/cols))
  print(f"rows: {rows}, cols: {cols}")
  #radar_title = f"Radars of the most representative person in each cluster"
  titles = [str(i) for i in cluster_labels_counts] # get cluster size as title
  specs = [[{'type': 'polar'}]*cols]*rows
  fig = make_subplots(rows=rows, cols=cols,
                            specs=specs,
                            horizontal_spacing=0.3/cols,
                            vertical_spacing=0.4/rows,
                            subplot_titles=titles,
                            )
  polar_args = {}
  for i in range(rows):
    for j in range(cols):
      #print(f"plotting: {i+1},{j+1}. Index = {i*cols+j+1}")
      if i*cols+j < medoids_df.shape[0]:
        if medoids_df.drop("cluster", axis=1).shape[1] == 5:
          r = medoids_df.iloc[i*cols+j]
          theta = ["IUsers", "EnvImpact", "CarShare", "TollsTraffic"]
        elif medoids_df.drop("cluster", axis=1).shape[1] == 6:
          r = medoids_df.drop("Country", axis=1).iloc[i*cols+j]
          theta = ["IUsers", "EnvImpact", "CarShare", "TollsTraffic", "Residence"]
        fig, polar_args = plot_individual_radar(fig, polar_args, 
                                                r, theta, 
                                                index=i*cols+j+1,
                                                row=i+1,col=j+1)
  fig.update_layout(
    height=row_height*rows,
    **polar_args)
  
  # Fixing sublot title positioning:
  # https://stackoverflow.com/questions/65775407/can-you-alter-a-subplot-title-location-in-plotly
  fig.update_annotations(patch=dict(yshift=20)) 
  fig.show()

  if save_path:
    fig.write_html(save_path)