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
        
        html.Div(
    [
        html.Div("Label Color Map:", style={
            "fontWeight": "bold",
            "color": "white",
            "marginBottom": "10px",
            "fontSize": "18px"
        }),
        html.Div([
            html.Span(style={"display": "inline-block", "width": "20px", "height": "20px", "backgroundColor": "black", "marginRight": "5px", "border": "1px solid white"}),
            html.Span("Nothing", style={"color": "white", "marginRight": "20px"}),

            html.Span(style={"display": "inline-block", "width": "20px", "height": "20px", "backgroundColor": "orange", "marginRight": "5px", "border": "1px solid white"}),
            html.Span("Up", style={"color": "white", "marginRight": "20px"}),

            html.Span(style={"display": "inline-block", "width": "20px", "height": "20px", "backgroundColor": "purple", "marginRight": "5px", "border": "1px solid white"}),
            html.Span("Left", style={"color": "white", "marginRight": "20px"}),

            html.Span(style={"display": "inline-block", "width": "20px", "height": "20px", "backgroundColor": "blue", "marginRight": "5px", "border": "1px solid white"}),
            html.Span("Right", style={"color": "white", "marginRight": "20px"}),

            html.Span(style={"display": "inline-block", "width": "20px", "height": "20px", "backgroundColor": "green", "marginRight": "5px", "border": "1px solid white"}),
            html.Span("Down", style={"color": "white", "marginRight": "20px"}),

            
        ])
    ],
    style={
        "padding": "15px",
        "backgroundColor": "rgba(30, 30, 30, 0.9)",  # Slightly transparent dark box
        "border": "1px solid white",
        "borderRadius": "8px",
        "marginBottom": "15px",
        "maxWidth": "fit-content"
    }
),

        
        
        
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

    # labels = np.random.randint(0,5,signal.shape[0])

    length_of_segment = 60
     
    for i in range(0,signal.shape[0],length_of_segment):
        randint = np.random.randint(0,10)
        if randint < 3: 
            
            labels[i:i+length_of_segment] = np.random.randint(1,5)
        
        
            
    
    
    
    
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

        //Slice windowed label 
        
        const time = Array.from({length: end - start}, (_, i) => i + start);
        const signal_window = signal.slice(start,end);
        const label_window = labels.slice(start,end);
        
        
        /*
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
        
        */
        
        
        const labelColorMap = {
            "0": "red",
            "1": "orange",
            "2": "purple",
            "3": "blue",
            "4": "green"
        };

        
        
        //Default to gray if unknown 
        
    
        
        const subplots = [];
        for (let ch = 0; ch < num_channels; ch++) {
            let segment_start = 0;
            while (segment_start < time.length) {
                const current_label = label_window[segment_start];
                let segment_end = segment_start + 1;

                while (
                    segment_end < time.length &&
                    label_window[segment_end] === current_label
                ) {
                    segment_end++;
                }

                const time_segment = time.slice(segment_start, segment_end);
                const y_segment = signal_window
                    .slice(segment_start, segment_end)
                    .map(row => row[ch]);

                subplots.push({
                    x: time_segment,
                    y: y_segment,
                    mode: 'lines',
                    name: `Ch ${ch + 1}`,
                    line: { color: labelColorMap[current_label] || 'black' },
                    xaxis: 'x',
                    yaxis: `y${ch + 1}`,
                    showlegend: false
                });

                segment_start = segment_end;
            }
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
