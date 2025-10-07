# PlayGround.py
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import math

# ===== Helpers for multi-plot rendering =====
def _make_single_fig(x, y, name):
    fig = go.Figure(
        data=[go.Scattergl(x=x, y=y, mode="lines", name=name, line=dict(width=1), hoverinfo="skip")]
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(title="muestras", showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(title="amplitud", showgrid=True, gridcolor="rgba(128,128,128,0.25)", zeroline=False, fixedrange=True),
        autosize=True,
    )
    return fig


def _demo_series(k=4, n=1000):
    x = list(range(n))
    out = []
    for i in range(k):
        y = [math.sin(2 * math.pi * (0.002 + 0.0007 * i) * t + i) for t in x]
        out.append({"x": x, "y": y, "name": f"demo-{i+1}"})
    return out


def render_plots_list(series, base_id="pg-plot", height_px=320):
    graphs = []
    for i, s in enumerate(series):
        name = s.get("name", f"series-{i}")
        fig = _make_single_fig(s["x"], s["y"], name)
        graphs.append(
            dcc.Graph(
                id=f"{base_id}-{i}",
                figure=fig,
                responsive=True,
                className="plot-item",
                style={"height": f"{height_px}px", "width": "100%", "minHeight": 0},
                config={
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "zoom",
                        "pan",
                        "select",
                        "lasso2d",
                        "zoomIn2d",
                        "zoomOut2d",
                        "autoScale2d",
                        "resetScale2d",
                        "toImage",
                    ],
                    "staticPlot": True,
                },
            )
        )
    return html.Div(id="plots-container", className="plots-list", children=graphs)

def get_playGround(title: str, description: str, metadata: dict, custom_metadata: dict | None = None, graph_id: str = "pg-main-plot", multi: bool = False, series: list | None = None, navigation_controls: html.Div | None = None):
    """
    Layout en 2 filas:
      - Fila 1: Metadata (izq) + Custom Metadata/Controles (der)
      - Fila 2: Plot (WebGL) dentro de un contenedor con SCROLL VERTICAL

    Args:
        navigation_controls: Componente opcional de controles para la tarjeta derecha
    """

    # ============== helpers UI ==============
    def class_chip(name: str, color: str):
        return html.Span(
            [
                html.Span("", style={
                    "display": "inline-block",
                    "width": "10px",
                    "height": "10px",
                    "borderRadius": "999px",
                    "marginRight": "8px",
                    "background": color or "var(--accent-2)",
                    "boxShadow": "0 0 0 2px color-mix(in srgb, var(--surface-0) 70%, transparent)"
                }),
                html.Span(name)
            ],
            className="class-chip",
            style={
                "display": "inline-flex",
                "alignItems": "center",
                "gap": "6px",
                "padding": "6px 10px",
                "borderRadius": "999px",
                "border": f"1px solid {color or 'var(--border-strong)'}",
                "background": "color-mix(in srgb, var(--surface-2) 70%, transparent)",
                "color": "var(--text)",
                "fontWeight": 600,
                "letterSpacing": "0.2px",
                "margin": "6px"
            }
        )

    def meta_grid_item(label: str, value: str):
        return html.Div(
            [
                html.Div(label, style={
                    "fontSize": "12px",
                    "opacity": 0.85,
                    "color": "var(--text-muted)"
                }),
                html.Div(value, style={
                    "fontSize": "16px",
                    "fontWeight": 700,
                    "color": "var(--text)"
                }),
            ],
            style={
                "padding": "10px 12px",
                "border": "1px solid var(--border-weak)",
                "borderRadius": "var(--radius-md)",
                "background": "var(--card-bg)",
                "boxShadow": "var(--shadow-1)"
            }
        )

    # ====== Metadata Unificada ======
    # Sección de chips de clases
    classes_section = []
    if metadata and isinstance(metadata, dict):
        for cls in sorted(metadata.keys()):
            col = metadata.get(cls) or "var(--accent-2)"
            classes_section.append(class_chip(cls, col))
    else:
        classes_section = [html.Div("Sin metadata de clases.", style={"opacity": 0.8, "color": "var(--text-muted)"})]

    # Datos numéricos de metadata
    ds_name = (custom_metadata or {}).get("dataset_name") or "—"
    n_cls   = str((custom_metadata or {}).get("num_classes", "—"))
    sfreq   = (custom_metadata or {}).get("sfreq")
    sfreq_s = f"{sfreq:.3f} Hz" if isinstance(sfreq, (int, float)) else "—"
    n_ch    = str((custom_metadata or {}).get("n_channels", "—"))
    unit    = (custom_metadata or {}).get("eeg_unit") or "—"

    # Grid unificado: clases arriba, datos abajo
    unified_metadata = html.Div(
        [
            # Sección de clases
            html.Div(
                [
                    html.Div("Clases", style={
                        "fontSize": "13px",
                        "fontWeight": "600",
                        "color": "var(--text-muted)",
                        "marginBottom": "12px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.5px"
                    }),
                    html.Div(classes_section, style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "8px",
                        "marginBottom": "24px"
                    })
                ]
            ),
            # Divisor
            html.Hr(style={
                "border": "none",
                "borderTop": "1px solid var(--border-weak)",
                "margin": "16px 0",
                "opacity": "0.6"
            }),
            # Grid de datos
            html.Div(
                [
                    meta_grid_item("Dataset", ds_name),
                    meta_grid_item("Número de clases", n_cls),
                    meta_grid_item("Frecuencia de muestreo", sfreq_s),
                    meta_grid_item("Canales", n_ch),
                    meta_grid_item("Unidad", unit),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                    "gap": "12px"
                }
            )
        ]
    )

    # ====== figura demo (WEBGL) ======
    N = 5000
    x = list(range(N))
    y = [math.sin(2*math.pi*0.005*i) + 0.5*math.sin(2*math.pi*0.011*i + 0.7) for i in x]
    fig_demo = go.Figure(
        data=[go.Scattergl(
            x=x, y=y, mode="lines", name="Demo EEG (WebGL)",
            line=dict(width=1), hoverinfo="skip"
        )]
    )
    fig_demo.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(title="muestras", showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(title="amplitud", showgrid=True, gridcolor="rgba(128,128,128,0.25)", zeroline=False, fixedrange=True),
        height=600  # altura inicial razonable; luego el clientside la ajusta dinámicamente
    )

    # ====== layout ======
    return dbc.Container(
        fluid=True,
        className="page-container pg-wrap",
        children=[
            html.H2(title + " PG", className="page-title"),
            html.P(description, className="page-description"),

            # Row 1: Metadata Unificada + Tarjeta vacía
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Metadata del Dataset", className="pg-card__title"),
                                html.Div(unified_metadata, className="pg-card__body"),
                            ],
                            className="pg-card pg-card--meta",
                        ),
                        width={"xs": 12, "md": 6},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Configuración", className="pg-card__title"),
                                html.Div(
                                    navigation_controls if navigation_controls else html.Div("Por definir", style={
                                        "opacity": 0.6,
                                        "color": "var(--text-muted)",
                                        "fontStyle": "italic",
                                        "padding": "24px",
                                        "textAlign": "center"
                                    }),
                                    className="pg-card__body"
                                ),
                            ],
                            className="pg-card pg-card--custom",
                        ),
                        width={"xs": 12, "md": 6},
                    ),
                ],
                className="mt-3 g-3",
                align="stretch",
            ),

            # Row 2: zona de plots (single o múltiples) con SCROLL local
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            children=[
                                # mantener un graph oculto para compatibilidad con callbacks existentes
                                dcc.Graph(
                                    id=graph_id,
                                    figure=fig_demo,
                                    responsive=True,
                                    style={"width": "100%", "minHeight": "460px", "display": "none"},
                                    config={
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": [
                                            "zoom","pan","select","lasso2d",
                                            "zoomIn2d","zoomOut2d","autoScale2d",
                                            "resetScale2d","toImage"
                                        ],
                                        "staticPlot": True
                                    },
                                ),
                                # lista de gráficos visibles si multi=True; si no, un único gráfico visible
                                (render_plots_list(series or _demo_series(), base_id=graph_id)
                                 if multi else
                                 dcc.Graph(
                                     id=f"{graph_id}-visible",
                                     figure=fig_demo,
                                     responsive=True,
                                     style={"width": "100%", "minHeight": "460px"},
                                     config={
                                         "displaylogo": False,
                                         "modeBarButtonsToRemove": [
                                             "zoom","pan","select","lasso2d",
                                             "zoomIn2d","zoomOut2d","autoScale2d",
                                             "resetScale2d","toImage"
                                         ],
                                         "staticPlot": True
                                     },
                                 ))
                            ],
                            className="plots-scroll-container",
                        ),
                        width=12,
                    ),
                ],
                className="mt-3",
                align="stretch",
            ),
        ],
    )
