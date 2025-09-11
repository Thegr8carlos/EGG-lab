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
from sklearn.preprocessing import LabelEncoder

from shared.fileUtils import get_Data_filePath
import os



#from shared.file_reader import is_data_loaded

# registrar página
register_page(__name__, path="/dataset", name="Dataset")

# layout con contenedor dinámico
layout = html.Div(
    children=[
        dcc.Store(id='full-signal-data'),
        dcc.Store(id='label-color-store'),

        html.Div(
    [
        html.Div("Label Color Map:", style={
            "fontWeight": "bold",
            "color": "white",
            "marginBottom": "10px",
            "fontSize": "18px"
        }),
        
        html.Div(
            id = "dynamic-color-legend",
            style = {
                "padding": "15px",
                "backgroundColor": "rgba(30, 30, 30, 0.9)",
                "border": "1px solid white",
                "borderRadius": "8px",
                "marginBottom": "15px",
                "maxWidth": "fit-content"
            }
        )
        
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
            interval=17,
           
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
    
    
    
    
    print(f"Selected path is: {selected_file_path}")
    
    #We check if file passed is valid 
    if not selected_file_path:
        
        
        print("Invalid or missing file.")
        return no_update, True  # Keep interval disabled
    
    
    #Now we check if file is a valid format
    if not selected_file_path.endswith(".npy"):
        
        #We check if there's a corresponding .npy in Aux 
        
        
        
        #We check that the path is actually valid as an absolute one of /Data
        if not os.path.exists(f"Data/{selected_file_path}"):
            return no_update, True

        mappedFilePath = get_Data_filePath(f"Data/{selected_file_path}")
        
        if os.path.exists(mappedFilePath):
            
            print(mappedFilePath)
            
            signal = np.load(mappedFilePath, mmap_mode = 'r')
            full_path = Path(mappedFilePath)
        else: 
            return no_update, True
            
    else:
        
        # Load the signal
        signal = np.load(f"Data/{selected_file_path}", mmap_mode = 'r')
        
        full_path =  Path(f"Data/{selected_file_path}")

        
    
    

    #we want to extract the parent path and the file name to obtain the label that is in the parent directory and in a folder named labels with the same file name 
    
    parentDir = full_path.parent
    fileName = full_path.name
    
    labels = np.load(f"{parentDir}/Labels/{fileName}", allow_pickle = True)
    if signal.shape[0] < signal.shape[1]: 
        signal = signal.T


    length_of_segment = 60
     
    for i in range(0,signal.shape[0],length_of_segment):
        randint = np.random.randint(0,10)
        if randint < 3: 
            
            labels[i:i+length_of_segment] = np.random.randint(1,5)
        
    labels = labels.astype(str)
    unique_labels = np.unique(labels)
    
    label_color_map  = {}
    
    for idx, label in enumerate(unique_labels):
        hue = (idx* 47) % 360
        label_color_map[str(label)] = f"hsl({hue}, 70%, 50%)"        

    
    
    
    
    

    '''
        Change this later when optimizing right now it's as is because the file wont load in time and the server times out, we need to optimize to load in chunks 
        or something like that, therefore we only load the first 50k points
    
    '''    
    
    signal = signal[:5000,:]
    labels = labels.reshape(-1)[:5000]
    
    print(f"signal shape: {signal.shape}")  
    print(f"labels shape: {labels.shape}")  
    
        
    # We encode the vector 
    
    encoder = LabelEncoder() 
    labels = encoder.fit_transform(labels.ravel())
    
    
    
    print(f"One hot encoded labels shape: {labels.shape}")
    
    
    
    
        
    # Serialize the signal (convert to list to make it JSON serializable)
    signal_dict = {
        "data": signal.tolist(),
        "num_channels": signal.shape[1],
        "num_timepoints": signal.shape[0], 
        "labels": labels.tolist(), 
        "label_color_map" : label_color_map
    }

    return signal_dict, False  # Enable interval


@callback(
    Output("dynamic-color-legend","children"), 
    Input("full-signal-data", "data")
)
def update_color_legend(signal_data): 
    
    if not signal_data or "label_color_map" not in signal_data:
        return "No label data loaded"
    
    color_map = signal_data["label_color_map"]
    
    legend_items = []   
    count = 0 
    for label, color in color_map.items(): 
        
        
        legend_items.append(
            html.Span(style = {
                "display": "inline-block",
                "width": "20px",
                "height": "20px",
                "backgroundColor": color,
                "marginRight": "5px",
                "border": "1px solid white"
            })
        )
        
        legend_items.append(
            html.Span(str(label), style = {
                "color" : "white", 
                "marginRight" : "20px"
            })
        )
        count +=1
        
        if count > 10:
            break
    return legend_items





clientside_callback(
    """
    
    
    function(n_intervals, signal_data) {
        
        
        if (!signal_data || !signal_data.data || !signal_data.labels) {
            return window.dash_clientside.no_update;
        }
        
        
        const labelColorMap = signal_data.label_color_map || {}; 
        
        
        function getColorForLabel(label) {
            return labelColorMap[label] || "gray";
        }
    
    
        
        
        
        const STEP = 1;
        const WINDOW = 100;
        const signal = signal_data.data;
        const labels = signal_data.labels;
        //console.log(labels);
        
        const num_channels = signal_data.num_channels;
        const num_timepoints = signal_data.num_timepoints;

        
        let visibleChannels = [];
        
        const container = document.getElementById("dashboard-scroll-wrapper-container");
        
        if (container){ 
            const scrollTop = container.scrollTop; 
            console.log(scrollTop);
            
            const containerHeight = container.clientHeight; 
            
            const channelHeight = 200; //Same as in layout 
            
            const firstVisibleChannel = Math.floor(scrollTop / channelHeight); 
            
            let numVisibleChannels = Math.ceil(containerHeight / channelHeight); 
            
            const lastVisibleChannel = Math.min(firstVisibleChannel + numVisibleChannels, num_channels);
            
            console.log(firstVisibleChannel); 
            console.log(lastVisibleChannel);
            
            visibleChannels = Array.from( {length: lastVisibleChannel - firstVisibleChannel}, (_,i) => i + firstVisibleChannel)
            
        }
        
        
        

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
        
        

        

        
        
        
        
    
        
        const subplots = [];
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
                const signal_segment = signal_window.slice(segment_start, segment_end);

                const color = getColorForLabel(current_label);
                
                
                for (let i=0; i < visibleChannels.length; i++){
                    const ch = visibleChannels[i];
                    
                    
                    const y_segment = signal_segment.map(row=>row[ch] || 0);
                
                
                    subplots.push({
                        x: time_segment,
                        y: y_segment,
                        mode: 'lines',
                        name: `Ch ${ch + 1}`,
                        line: { color: color || 'black' },
                        xaxis: 'x',
                        yaxis: `y${ch + 1}`,
                        showlegend: false
                    });
                }

                segment_start = segment_end;
            }


        const layout = {
            height: 200 * num_channels,
            showlegend: false,
            title: "Señales Multicanal (Desplazamiento Automático)",
            margin: {t: 40, b: 40},
        };



        const gap = 0;
        const totalGaps = gap * (num_channels-1);
        const plotHeight = (1 - totalGaps) / num_channels;

        


        for (let i = 0; i < num_channels; i++) {
            layout[`yaxis${i+1}`] = {
                title: `Ch ${i+1}`,
                domain: [
                    1 - (plotHeight + gap) * (i + 1),
                    1 - (plotHeight + gap) * i
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
