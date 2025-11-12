# components/dataset_view.py
from dash import html, dcc, clientside_callback, Output, Input, State, callback
from typing import Optional, Dict, Any
from app.components.DashBoard import get_dashboard_container_dynamic

def get_dataset_view(
    container_id: str = "dataset-view",
    full_signal_store_id: str = "full-signal-data",
    label_color_store_id: str = "label-color-store",
    legend_container_id: str = "dynamic-color-legend",
    graph_id: str = "signal-graph",
    interval_id: str = "interval-component",
    interval_ms: int = 17,
    outer_box_style: Optional[Dict[str, Any]] = None,
    inner_box_style: Optional[Dict[str, Any]] = None,
):
    default_outer_style = {
        "padding": "15px",
        "backgroundColor": "rgba(30, 30, 30, 0.9)",
        "border": "1px solid white",
        "borderRadius": "8px",
        "marginBottom": "15px",
        "maxWidth": "fit-content",
    }
    default_inner_style = {
        "padding": "15px",
        "backgroundColor": "rgba(30, 30, 30, 0.9)",
        "border": "1px solid white",
        "borderRadius": "8px",
        "marginBottom": "15px",
        "maxWidth": "fit-content",
    }

    outer_style = {**default_outer_style, **(outer_box_style or {})}
    inner_style = {**default_inner_style, **(inner_box_style or {})}

    return html.Div(
        id=container_id,
        children=[
            dcc.Store(id=full_signal_store_id),
            dcc.Store(id=label_color_store_id),

            html.Div(
                [
                    html.Div(
                        "Label Color Map:",
                        style={
                            "fontWeight": "bold",
                            "color": "white",
                            "marginBottom": "10px",
                            "fontSize": "18px",
                        },
                    ),
                    html.Div(id=legend_container_id, style=inner_style),
                ],
                style=outer_style,
            ),

            get_dashboard_container_dynamic(graph_id=graph_id),

            dcc.Interval(
                id=interval_id,
                interval=interval_ms,
                n_intervals=0,
                disabled=True,
            ),
        ],
    )


# --- REGISTRADORES DE CALLBACKS (parametrizados por ID) ---

def register_dataset_clientside(graph_id: str, interval_id: str, store_id: str) -> None:
    """Registra el clientside_callback del gráfico con IDs dinámicos."""
    clientside_callback(
        """
        function(n_intervals, signal_data) {
            if (!signal_data || !signal_data.data || !signal_data.labels) {
                return window.dash_clientside.no_update;
            }

            const labelColorMap = signal_data.label_color_map || {};
            const getColorForLabel = (label) => labelColorMap[label] || "gray";

            const STEP = 1;
            const WINDOW = 100;
            const signal = signal_data.data;
            const labels = signal_data.labels;
            const num_channels = signal_data.num_channels;
            const num_timepoints = signal_data.num_timepoints;

            // Extract metadata
            const metadata = signal_data.metadata || {};
            const channel_names = metadata.channel_names || [];
            const sampling_freq = metadata.sampling_frequency_hz;
            const dataset_name = metadata.dataset_name;
            const eeg_unit = metadata.eeg_unit || "V";

            let start = n_intervals * STEP;
            let end = start + WINDOW;
            if (end > num_timepoints) { start = 0; end = WINDOW; }

            const time = Array.from({length: end - start}, (_, i) => i + start);
            const signal_window = signal.slice(start, end);
            const label_window  = labels.slice(start, end);

            const traces = [];
            let segment_start = 0;

            while (segment_start < time.length) {
                const current_label = label_window[segment_start];
                let segment_end = segment_start + 1;
                while (segment_end < time.length && label_window[segment_end] === current_label) {
                    segment_end++;
                }

                const time_segment = time.slice(segment_start, segment_end);
                const signal_segment = signal_window.slice(segment_start, segment_end);
                const color = getColorForLabel(current_label);

                for (let ch = 0; ch < num_channels; ch++) {
                    const y_segment = signal_segment.map(row => (!row || row.length <= ch) ? 0 : row[ch]);
                    const channel_name = channel_names[ch] || `Ch ${ch + 1}`;
                    traces.push({
                        x: time_segment,
                        y: y_segment,
                        mode: 'lines',
                        name: channel_name,
                        line: { color: color || 'black' },
                        xaxis: 'x',
                        yaxis: `y${ch + 1}`,
                        showlegend: false
                    });
                }
                segment_start = segment_end;
            }

            // Build title with metadata info
            let title = "Señales Multicanal (Desplazamiento Automático)";
            if (dataset_name) {
                title += `<br><sub>Dataset: ${dataset_name}`;
                if (sampling_freq) {
                    title += ` | Fs: ${sampling_freq} Hz`;
                }
                if (eeg_unit) {
                    title += ` | Unidad: ${eeg_unit}`;
                }
                title += `</sub>`;
            }

            const layout = {
                height: 200 * num_channels,
                showlegend: false,
                title: title,
                margin: {t: 80, b: 40}
            };
            for (let i = 0; i < num_channels; i++) {
                const channel_name = channel_names[i] || `Ch ${i+1}`;
                layout[`yaxis${i+1}`] = {
                    title: channel_name,
                    domain: [1 - (i + 1) / num_channels, 1 - i / num_channels]
                };
            }

            return { data: traces, layout: layout };
        }
        """,
        Output(graph_id, "figure"),
        [Input(interval_id, "n_intervals"), Input(store_id, "data")],
    )


def register_dataset_legend(legend_container_id: str, store_id: str) -> None:
    """Registra el callback del mapa de colores (server-side) con IDs dinámicos."""
    @callback(
        Output(legend_container_id, "children"),
        Input(store_id, "data"),
    )
    def _update_color_legend(signal_data):
        if not signal_data or "label_color_map" not in signal_data:
            return "No label data loaded"

        color_map = signal_data["label_color_map"]
        legend_items = []
        for idx, [label, color] in enumerate(color_map.items()):
            legend_items.append(
                html.Span(style={
                    "display": "inline-block",
                    "width": "20px",
                    "height": "20px",
                    "backgroundColor": color,
                    "marginRight": "5px",
                    "border": "1px solid white"
                })
            )
            legend_items.append(
                html.Span(str(label), style={"color": "white", "marginRight": "20px"})
            )
            if idx >= 9:
                break
        return legend_items