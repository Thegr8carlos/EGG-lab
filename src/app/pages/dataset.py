from pathlib import Path
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
            interval=1,
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
    

    #we want to extract the parent path and the file name to obtain the label that is in the parent directory and in a folder named labels with the same file name 
    
    full_path =  Path(f"Data/{selected_file_path}")
    parentDir = full_path.parent
    fileName = full_path.name
    
    labels = np.load(f"{parentDir}/Labels/{fileName}", allow_pickle = True)

    if signal.shape[0] < signal.shape[1]: 
        signal = signal.T


    
    print(f"signal shape: {signal.shape}")  
    print(f"labels shape: {labels.shape}")  

    # Serialize the signal (convert to list to make it JSON serializable)
    signal_dict = {
        "data": signal.tolist(),
        "num_channels": signal.shape[1],
        "num_timepoints": signal.shape[0], 
        "labels": labels.tolist()
    }

    return signal_dict, False  # Enable interval



clientside_callback(
    """
    function(n_intervals, signal_data) {
        if (!signal_data || !signal_data.data || !signal_data.labels) {
            return window.dash_clientside.no_update;
        }
        const STEP = 10;
        const WINDOW = 1000;
        const signal = signal_data.data;
        const labels = signal_data.labels;
        console.log(labels);
        const num_channels = signal_data.num_channels;
        const num_timepoints = signal_data.num_timepoints;

        let start = n_intervals * STEP;
        let end = start + WINDOW;

        if (end > num_timepoints) {
            start = 0;
            end = WINDOW;
        }

        const time = Array.from({length: end - start}, (_, i) => i + start);
        //Slice windowed label 
        
        const label_window = labels.slice(start,end);
        
        const labelColorMap = {
            "Comment/Down": "red",
            "Comment/Left": "orange",
            "Comment/Rest": "purple",
            "Comment/Right": "blue",
            "Comment/Select": "green",
            "Comment/Up": "brown",
            "Comment/Wait": "pink",
            "Comment/WarmUp": "teal",
            "None": "gray"
        };

        
        
        //Default to gray if unknown 
        
        const mid_label = label_window[Math.floor(label_window.length / 2)];
        const color = labelColorMap[mid_label] || 'black';
        
        
        const subplots = [];
        for (let i = 0; i < num_channels; i++) {
            const channelData = signal.slice(start, end).map(row => row[i]);
            subplots.push({
                x: time,
                y: channelData,
                mode: 'lines',
                name: `Channel ${i+1}`,
                yaxis: `y${i+1}`,
                xaxis: 'x',
                line: {color: color}
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
