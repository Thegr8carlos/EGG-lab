import plotly.express as px
from dash import html, dcc, register_page, callback, Output, Input, no_update, State, MATCH, clientside_callback
import plotly.express as px
import numpy as np

from app.components.PageContainer import get_page_container
from app.components.DashBoard import get_dashboard_container, get_dashboard_container_dynamic

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash


#from shared.file_reader import is_data_loaded

# registrar página
register_page(__name__, path="/dataset", name="Dataset")

# layout con contenedor dinámico
layout = html.Div(
    children=[
        dcc.Store(id='full-signal-data'),
        get_dashboard_container_dynamic(graph_id = "signal-graph"),
        
        dcc.Interval(
            id='interval-component',
            interval=10,
            n_intervals=0,
            disabled=True
        )
    ],
    id="dataset-view"
)







@callback(
    Output("full-signal-data", "data"),
    Output("interval-component", "disabled"),
    Input("selected-file-path", "data")
)
def load_signal_data(selected_file_path):
    print(selected_file_path)
    if not selected_file_path or not selected_file_path.endswith(".npy"):
        print("Invalid or missing file.")
        return no_update, True  # Keep interval disabled
    
    # Load the signal
    signal = np.load(f"Data/{selected_file_path}")

    if signal.shape[0] < signal.shape[1]: 
        signal = signal.T

    # Serialize the signal (convert to list to make it JSON serializable)
    signal_dict = {
        "data": signal.tolist(),
        "num_channels": signal.shape[1],
        "num_timepoints": signal.shape[0]
    }

    return signal_dict, False  # Enable interval



# @callback(
#     Output("signal-graph", "figure"),
#     Input("interval-component", "n_intervals"),
#     State("full-signal-data", "data"), 
#     prevent_initial_call = True
# )
# def update_signal_graph(n_intervals, signal_data):
    
    
#     STEP = 10 
#     WINDOW = 1000
    
#     if not signal_data:
#         raise dash.exceptions.PreventUpdate

#     signal = np.array(signal_data["data"])
#     num_channels = signal.shape[1]
#     num_timepoints = signal.shape[0]

#     start = n_intervals * STEP
#     end = start + WINDOW

#     if end > num_timepoints:
#         start = 0
#         end = WINDOW

#     time = np.arange(start, end)

#     fig = make_subplots(
#         rows=num_channels,
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.02,
#         subplot_titles=[f"Channel {i+1}" for i in range(num_channels)]
#     )

#     for i in range(num_channels):
#         fig.add_trace(
#             go.Scatter(
#                 x=time,
#                 y=signal[start:end, i],
#                 mode="lines",
#                 name=f"Channel {i+1}"
#             ),
#             row=i+1,
#             col=1
#         )
#         fig.update_yaxes(title_text= f"Ch {i+1}", row=i+1,col=1)    
    
   

#     fig.update_layout(
#         height=250 * num_channels,
#         showlegend=False,
#         title="Señales Multicanal (Desplazamiento Automático)",
#         margin=dict(t=40, b=40),
#         # xaxis=dict(range=[start, end])
#     )

    
#     return fig


clientside_callback(
    """
    function(n_intervals, signal_data) {
        if (!signal_data || !signal_data.data) {
            return window.dash_clientside.no_update;
        }

        const STEP = 10;
        const WINDOW = 1000;
        const signal = signal_data.data;
        const num_channels = signal_data.num_channels;
        const num_timepoints = signal_data.num_timepoints;

        let start = n_intervals * STEP;
        let end = start + WINDOW;

        if (end > num_timepoints) {
            start = 0;
            end = WINDOW;
        }

        const time = Array.from({length: end - start}, (_, i) => i + start);

        const subplots = [];
        for (let i = 0; i < num_channels; i++) {
            const channelData = signal.slice(start, end).map(row => row[i]);
            subplots.push({
                x: time,
                y: channelData,
                mode: 'lines',
                name: `Channel ${i+1}`,
                yaxis: `y${i+1}`,
                xaxis: 'x'
            });
        }

        const layout = {
            height: 250 * num_channels,
            showlegend: false,
            title: "Señales Multicanal (Desplazamiento Automático)",
            margin: {t: 40, b: 40},
        };

        for (let i = 0; i < num_channels; i++) {
            layout[`yaxis${i+1}`] = {
                title: `Ch ${i+1}`,
                domain: [
                    1 - (i + 1) / num_channels,
                    1 - i / num_channels
                ]
            };
        }

        return {
            data: subplots,
            layout: layout
        };
    }
    """,
    Output("signal-graph", "figure"),
    Input("interval-component", "n_intervals"),
    State("full-signal-data", "data")
)
