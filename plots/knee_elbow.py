import plotly.graph_objects as go

def PlotElbow(wss_values, k_values, title="", save_path=None):

    k_values = list(k_values)
    fig = go.Figure(data=go.Scatter(x=k_values,
                                    y=wss_values,
                                    mode='lines+markers'))
    fig.update_layout(title=title)
    fig.show()

    if save_path:
        fig.write_html(save_path)