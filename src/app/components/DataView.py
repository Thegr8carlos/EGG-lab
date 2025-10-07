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
    # ---------- Estilos (solo presentación) ----------
    CENTER_WRAP = {
        "display": "flex",
        "justifyContent": "center",
        "width": "100%",
        "padding": "0 0.5rem",
    }
    CENTER_COL = {
        # columna central: leyenda arriba (compacta), plot abajo (protagónico)
        "width": "min(1700px, 95vw)",          # ⬅️ más ancho
        "display": "grid",
        "gridTemplateRows": "auto minmax(78vh, auto)",  # ⬅️ plot más alto
        "gap": "1rem",
        "alignItems": "start",
        "margin": "0 auto",
    }
    CARD_BASE = {
        "borderRadius": "16px",
        "padding": "1rem",
        "boxShadow": "0 10px 24px rgba(0,0,0,0.30)",
        "border": "1px solid",
        "borderColor": "color-mix(in srgb, var(--color-4) 35%, transparent)",
        "background": "linear-gradient(180deg, color-mix(in srgb, var(--color-2) 85%, transparent), var(--color-1))",
    }
    # Leyenda más pequeña
    LEGEND_CARD = {
        **CARD_BASE,
        "padding": "0.6rem 0.8rem",   # ⬅️ compacto
        "minHeight": "60px",
    }
    PLOT_CARD = {
        **CARD_BASE,
        "overflowX": "auto",
        "minHeight": "78vh",          # ⬅️ alto por defecto
        "padding": "1rem 1.25rem",
    }
    HEADER_TXT = {
        "fontWeight": 700,
        "letterSpacing": "0.2px",
        "marginBottom": "0.4rem",
        "color": "var(--color-3)",
    }
    SUBTEXT = {
        "opacity": 0.8,
        "fontSize": "0.88rem",        # ⬅️ texto de ayuda más chico
        "margin": "0 0 0.5rem 0",
        "color": "var(--color-3)",
    }
    LEGEND_WRAP = {
        "display": "flex",
        "flexWrap": "wrap",
        "gap": "6px 12px",            # ⬅️ más apretado
        "alignItems": "center",
        "padding": "6px",
        "borderRadius": "10px",
        "backgroundColor": "rgba(255,255,255,0.02)",
        "border": "1px dashed color-mix(in srgb, var(--color-5) 45%, transparent)",
        "color": "var(--color-3)",
        "fontSize": "0.9rem",         # ⬅️ etiquetas más pequeñas
    }

    return html.Div(
        id=container_id,
        children=[
            # Stores (NO se tocan)
            dcc.Store(id=full_signal_store_id),
            dcc.Store(id=label_color_store_id),

            # Columna central: leyenda ARRIBA (compacta), plot ABAJO (grande)
            html.Div(
                style=CENTER_WRAP,
                children=html.Div(
                    style=CENTER_COL,
                    children=[
                        # --- Leyenda arriba ---
                        html.Div(
                            style=LEGEND_CARD,
                            children=[
                                html.Div("Etiquetas y mapa de color", style=HEADER_TXT),
                                html.Div(
                                    "Consulta las clases y sus colores asignados al stream de EEG.",
                                    style=SUBTEXT
                                ),
                                html.Div(id=legend_container_id, style=LEGEND_WRAP),
                            ],
                        ),
                        # --- Plot centrado y mucho más grande ---
                        html.Div(
                            style=PLOT_CARD,
                            children=[
                                html.Div("Señal multicanal (plot dinámico)", style=HEADER_TXT),
                                html.Div(
                                    # NO tocamos IDs ni lógica interna del gráfico
                                    get_dashboard_container_dynamic(graph_id=graph_id),
                                    # el contenedor empuja a ocupar ancho y alto
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "alignItems": "stretch",
                                        "minHeight": "74vh",   # ⬅️ empuja al alto
                                        "width": "100%",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ),

            # Interval (NO se toca)
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
            const channel_names = signal_data.channel_names || [];

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
                    traces.push({
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

            // ---- Layout base ----
            const layout = {
                height: 200 * num_channels,
                showlegend: false,
                title: "Señales Multicanal (Desplazamiento Automático)",
                margin: {t: 40, b: 40, l: 60, r: 20}
            };

            // ---- Sub-ejes + anotaciones persistentes por canal ----
            const annotations = [];
            for (let i = 0; i < num_channels; i++) {
                const domainStart = 1 - (i + 1) / num_channels;
                const domainEnd   = 1 - i / num_channels;

                layout[`yaxis${i+1}`] = {
                    title: `Ch ${i+1}`,
                    domain: [domainStart, domainEnd]
                };

                // Nombre del canal y último valor visible en ventana (si existe)
                let lastVal = null;
                if (signal_window.length > 0) {
                    const lastRow = signal_window[signal_window.length - 1];
                    if (lastRow && lastRow.length > i) lastVal = lastRow[i];
                }
                const chName = (channel_names[i] || `Ch ${i+1}`);
                const label  = (lastVal !== null && lastVal !== undefined)
                    ? `${chName} • ${lastVal.toFixed(5)}`
                    : chName;

                // Anotación en la esquina superior-derecha de cada subtrama (coordenadas "paper")
                annotations.push({
                    xref: 'paper', yref: 'paper',
                    x: 0.985, y: (domainEnd - 0.015),  // un poco por debajo del borde superior del dominio
                    xanchor: 'right', yanchor: 'top',
                    text: label,
                    showarrow: false,
                    align: 'right',
                    font: { size: 12, color: '#e6e6e6' },
                    bgcolor: 'rgba(0,0,0,0.35)',
                    bordercolor: 'rgba(255,255,255,0.18)',
                    borderwidth: 1,
                    borderpad: 2
                });
            }
            layout.annotations = annotations;

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