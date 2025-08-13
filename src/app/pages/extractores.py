import os
from pathlib import Path
import numpy as np
from dash import html, dcc, register_page, callback, Output, Input, State, clientside_callback, no_update
from sklearn.preprocessing import LabelEncoder

from app.components.DashBoard import get_dashboard_container_dynamic
from shared.fileUtils import get_Data_filePath

# Página
register_page(__name__, path="/extractores", name="Transformadas")

layout = html.Div(
    id="transformadas-view",
    children=[
        # Stores
        dcc.Store(id="full-signal-data-transform"),
        dcc.Store(id="label-color-store-transform"),

        # Controles
        html.Div(
            [
                html.Div("Visualización de Transformadas", style={
                    "fontWeight": "bold",
                    "color": "white",
                    "fontSize": "18px",
                    "marginBottom": "8px",
                }),
                html.Div(
                    [
                        html.Label("Transformada:", style={"color": "white", "marginRight": "8px"}),
                        dcc.Dropdown(
                            id="transform-type",
                            options=[
                                {"label": "FFT", "value": "fft"},
                                {"label": "Haar (demo)", "value": "haar"},
                            ],
                            value="fft",
                            clearable=False,
                            style={"width": "220px"},
                        ),
                        html.Span(" | Ventana: 100 muestras", style={"color": "#bbb", "marginLeft": "16px"}),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "8px"}
                ),
            ],
            style={
                "padding": "12px 16px",
                "backgroundColor": "rgba(30, 30, 30, 0.9)",
                "border": "1px solid white",
                "borderRadius": "8px",
                "marginBottom": "12px",
                "maxWidth": "fit-content",
            },
        ),

        # Leyenda de colores (igual que dataset)
        html.Div(
            [
                html.Div(
                    "Label Color Map:",
                    style={
                        "fontWeight": "bold",
                        "color": "white",
                        "marginBottom": "10px",
                        "fontSize": "16px",
                    },
                ),
                html.Div(
                    id="dynamic-color-legend-transform",
                    style={
                        "padding": "10px",
                        "backgroundColor": "rgba(30, 30, 30, 0.9)",
                        "border": "1px solid white",
                        "borderRadius": "8px",
                        "marginBottom": "15px",
                        "maxWidth": "fit-content",
                    },
                ),
            ],
            style={
                "padding": "12px",
                "backgroundColor": "rgba(30, 30, 30, 0.9)",
                "border": "1px solid white",
                "borderRadius": "8px",
                "marginBottom": "15px",
                "maxWidth": "fit-content",
            },
        ),

        # Mismo componente de plot, id distinto
        get_dashboard_container_dynamic(graph_id="signal-graph-transform"),

        dcc.Interval(
            id="interval-component-transform",
            interval=17,
            n_intervals=0,
            disabled=True,
        ),
    ],
)

# ------- Callbacks (carga de datos, como en dataset) -------

@callback(
    Output("full-signal-data-transform", "data"),
    Output("interval-component-transform", "disabled"),
    Input("selected-file-path", "data"),
)
def load_signal_data_transform(selected_file_path):
    if not selected_file_path:
        return no_update, True

    if not selected_file_path.endswith(".npy"):
        if not os.path.exists(f"Data/{selected_file_path}"):
            return no_update, True
        mappedFilePath = get_Data_filePath(f"Data/{selected_file_path}")
        if os.path.exists(mappedFilePath):
            signal = np.load(mappedFilePath, mmap_mode="r")
            full_path = Path(mappedFilePath)
        else:
            return no_update, True
    else:
        signal = np.load(f"Data/{selected_file_path}", mmap_mode="r")
        full_path = Path(f"Data/{selected_file_path}")

    parentDir = full_path.parent
    fileName = full_path.name
    labels = np.load(f"{parentDir}/Labels/{fileName}", allow_pickle=True)

    if signal.shape[0] < signal.shape[1]:
        signal = signal.T

    # Segmentos demo (como dataset)
    length_of_segment = 60
    for i in range(0, signal.shape[0], length_of_segment):
        if np.random.randint(0, 10) < 3:
            labels[i:i + length_of_segment] = np.random.randint(1, 5)
    labels = labels.astype(str)

    unique_labels = np.unique(labels)
    label_color_map = {}
    for idx, label in enumerate(unique_labels):
        hue = (idx * 47) % 360
        label_color_map[str(label)] = f"hsl({hue}, 70%, 50%)"

    # Recorte por performance
    signal = signal[:5000, :]
    labels = labels.reshape(-1)[:5000]

    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels.ravel())

    signal_dict = {
        "data": signal.tolist(),
        "num_channels": signal.shape[1],
        "num_timepoints": signal.shape[0],
        "labels": labels.tolist(),
        "label_color_map": label_color_map,
    }
    return signal_dict, False


@callback(
    Output("dynamic-color-legend-transform", "children"),
    Input("full-signal-data-transform", "data"),
)
def update_color_legend_transform(signal_data):
    if not signal_data or "label_color_map" not in signal_data:
        return "No label data loaded"

    color_map = signal_data["label_color_map"]
    legend_items = []
    count = 0
    for label, color in color_map.items():
        legend_items.append(
            html.Span(
                style={
                    "display": "inline-block",
                    "width": "16px",
                    "height": "16px",
                    "backgroundColor": color,
                    "marginRight": "6px",
                    "border": "1px solid white",
                }
            )
        )
        legend_items.append(
            html.Span(str(label), style={"color": "white", "marginRight": "16px"})
        )
        count += 1
        if count > 12:
            break
    return legend_items


# --------- Clientside: mitad tiempo / mitad transformada ---------
clientside_callback(
    """
    function(n_intervals, transform_type, signal_data) {
        if (!signal_data || !signal_data.data || !signal_data.labels) {
            return window.dash_clientside.no_update;
        }

        const labelColorMap = signal_data.label_color_map || {};
        const getColorForLabel = (l) => labelColorMap[l] || "gray";

        // Streaming
        const STEP = 1;      // 1 muestra por tick
        const WINDOW = 100;  // tamaño de ventana visible
        const signal = signal_data.data;
        const labels = signal_data.labels;
        const C = signal_data.num_channels;
        const T = signal_data.num_timepoints;

        // Ventana ACTUAL (para ambos paneles)
        let start = n_intervals * STEP;
        let end   = start + WINDOW;
        if (end > T) { start = 0; end = WINDOW; }

        const tCurr  = Array.from({length: end - start}, (_, i) => i + start);
        const sigCurr = signal.slice(start, end);
        const labCurr = labels.slice(start, end);

        // Color “principal” de la ventana actual (para la transformada)
        const specColor = getColorForLabel(labCurr[labCurr.length - 1]);

        // --- FFT naive (N=100) ---
        function fftMagReal(x) {
            const N = x.length;
            const half = Math.floor(N/2);
            const mags = new Array(half);
            for (let k = 0; k < half; k++) {
                let re = 0, im = 0;
                for (let n = 0; n < N; n++) {
                    const phi = -2 * Math.PI * k * n / N;
                    re += x[n] * Math.cos(phi);
                    im += x[n] * Math.sin(phi);
                }
                mags[k] = Math.sqrt(re*re + im*im) / N;
            }
            return mags;
        }

        // --- Haar demo (3 niveles, energía) ---
        function haarDemo(x) {
            let arr = x.slice();
            const energies = [];
            for (let level = 0; level < 3; level++) {
                const a = [], d = [];
                for (let i = 0; i < arr.length - 1; i += 2) {
                    const approx = (arr[i] + arr[i+1]) / Math.SQRT2;
                    const detail = Math.abs((arr[i] - arr[i+1]) / Math.SQRT2);
                    a.push(approx); d.push(detail);
                }
                const L = Math.floor(x.length/2);
                const band = new Array(L).fill(0);
                for (let i = 0; i < L; i++) {
                    const src = Math.floor(i * d.length / L);
                    band[i] = d[Math.min(src, d.length-1)] || 0;
                }
                energies.push(band);
                arr = a;
                if (a.length < 2) break;
            }
            const L = Math.floor(x.length/2);
            const out = new Array(L).fill(0);
            for (let b = 0; b < energies.length; b++) {
                for (let i = 0; i < L; i++) out[i] = Math.max(out[i], energies[b][i] || 0);
            }
            return out;
        }

        // Layout con DOS ejes Y por canal:
        //  - yaxis(1..C): izquierda (crudo), anchor 'x'
        //  - yaxis(C+1..2C): derecha (transformada), anchor 'x2'
        const layout = {
            height: 200 * C,
            showlegend: false,
            title: "Señal cruda (izq)  vs  Transformada (der, ventana actual)",
            margin: {t: 40, b: 40, l: 60, r: 20},
            xaxis:  { domain: [0.0, 0.48], title: "Tiempo (muestras)" },
            xaxis2: { domain: [0.52, 1.0], title: (transform_type === "fft" ? "Frecuencia (bins)" : "Energía Haar") },
        };

        for (let i = 0; i < C; i++) {
            const domLow  = 1 - (i + 1) / C;
            const domHigh = 1 - i / C;
            // Izquierda (crudo)
            layout[`yaxis${i+1}`] = {
                title: `Ch ${i+1}`,
                domain: [domLow, domHigh],
                anchor: 'x',
                autorange: true
            };
            // Derecha (transformada)
            layout[`yaxis${C + i + 1}`] = {
                title: `Ch ${i+1}`,
                domain: [domLow, domHigh],
                anchor: 'x2',
                autorange: true
            };
        }

        const traces = [];

        // --- IZQUIERDA: señal cruda (misma ventana actual), segmentada por etiqueta ---
        let segStart = 0;
        while (segStart < tCurr.length) {
            const segLabel = labCurr[segStart];
            let segEnd = segStart + 1;
            while (segEnd < tCurr.length && labCurr[segEnd] === segLabel) segEnd++;
            const time_seg = tCurr.slice(segStart, segEnd);
            const sig_seg  = sigCurr.slice(segStart, segEnd);
            const color = getColorForLabel(segLabel);

            for (let ch = 0; ch < C; ch++) {
                const y_seg = sig_seg.map(row => (row && row.length > ch) ? row[ch] : 0);
                traces.push({
                    x: time_seg,
                    y: y_seg,
                    mode: 'lines',
                    line: { color: color || 'black' },
                    xaxis: 'x',
                    yaxis: `y${ch + 1}`,           // <- eje Y de la IZQUIERDA (solo crudo)
                    showlegend: false,
                    hoverinfo: 'x+y',
                });
            }
            segStart = segEnd;
        }

        // --- DERECHA: transformada de la ventana ACTUAL ---
        for (let ch = 0; ch < C; ch++) {
            const x_curr = sigCurr.map(row => (row && row.length > ch) ? row[ch] : 0);
            const spec = (transform_type === "fft") ? fftMagReal(x_curr) : haarDemo(x_curr);
            const fBins = Array.from({length: spec.length}, (_, i) => i);

            traces.push({
                x: fBins,
                y: spec,
                mode: 'lines',
                line: { color: specColor || 'gray' },
                xaxis: 'x2',
                yaxis: `y${C + ch + 1}`,          // <- eje Y de la DERECHA (solo transformada)
                showlegend: false,
                hoverinfo: 'x+y',
            });
        }

        return { data: traces, layout: layout };
    }
    """,
    Output("signal-graph-transform", "figure"),
    [Input("interval-component-transform", "n_intervals"),
     Input("transform-type", "value")],
    State("full-signal-data-transform", "data"),
)
