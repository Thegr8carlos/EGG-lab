# PlayGround.py
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import math

def get_playGround(title: str, description: str, metadata: dict, custom_metadata: dict, graph_id: str = "pg-main-plot"):
    """
    Layout en 2 filas:
      - Fila 1: Metadata (izq) + Custom Metadata (der)
      - Fila 2: Plot (WebGL, sin eventos)

    Args:
      metadata: dict { clase -> color_css }
      custom_metadata: dict con keys:
        dataset_name, num_classes, sfreq, n_channels, eeg_unit
      graph_id: id único para el dcc.Graph (permite reutilizar el componente sin colisiones)
    """

    # ============== helpers UI ==============
    def class_chip(name: str, color: str):
        # chip con puntito de color y borde del mismo color
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

    # ====== contenido tarjeta Metadata (clases) ======
    classes_section = []
    if metadata and isinstance(metadata, dict):
        for cls in sorted(metadata.keys()):
            col = metadata.get(cls) or "var(--accent-2)"
            classes_section.append(class_chip(cls, col))
    else:
        classes_section = [html.Div("Sin metadata de clases.", style={"opacity": 0.8})]

    # ====== contenido tarjeta Custom Metadata ======
    ds_name = (custom_metadata or {}).get("dataset_name") or "—"
    n_cls   = str((custom_metadata or {}).get("num_classes", "—"))
    sfreq   = (custom_metadata or {}).get("sfreq")
    sfreq_s = f"{sfreq:.3f} Hz" if isinstance(sfreq, (int, float)) else "—"
    n_ch    = str((custom_metadata or {}).get("n_channels", "—"))
    unit    = (custom_metadata or {}).get("eeg_unit") or "—"

    custom_grid = html.Div(
        [
            meta_grid_item("Dataset", ds_name),
            meta_grid_item("Clases", n_cls),
            meta_grid_item("Fs", sfreq_s),
            meta_grid_item("Canales", n_ch),
            meta_grid_item("Unidad", unit),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
            "gap": "12px"
        }
    )

    # ====== figura demo (WEBGL, fija y sin eventos) ======
    N = 5000
    x = list(range(N))
    y = [math.sin(2*math.pi*0.005*i) + 0.5*math.sin(2*math.pi*0.011*i + 0.7) for i in x]

    fig_demo = go.Figure(
        data=[go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            name="Demo EEG (WebGL)",
            line=dict(width=1),
            hoverinfo="skip"
        )]
    )
    fig_demo.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(title="muestras", showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(title="amplitud", showgrid=True, gridcolor="rgba(128,128,128,0.25)", zeroline=False, fixedrange=True),
    )

    # ====== layout ======
    return dbc.Container(
        fluid=True,
        className="page-container pg-wrap",
        children=[
            html.H2(title + " PG", className="page-title"),
            html.P(description, className="page-description"),

            # Row 1: Metadata + Custom Metadata
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Metadata", className="pg-card__title"),
                                html.Div(classes_section, className="pg-card__body"),
                            ],
                            className="pg-card pg-card--meta",
                        ),
                        width={"xs": 12, "md": 6},
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Custom Metadata", className="pg-card__title"),
                                html.Div(custom_grid, className="pg-card__body"),
                            ],
                            className="pg-card pg-card--custom",
                        ),
                        width={"xs": 12, "md": 6},
                    ),
                ],
                className="mt-3 g-3",
                align="stretch",
            ),

            # Row 2: Plot grande
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.Div("Plot (WebGL – clientside)", className="pg-card__title"),
                                html.Div(
                                    dcc.Graph(
                                        id=graph_id,  # id parametrizable para evitar colisiones entre páginas
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
                                    ),
                                    className="pg-card__body pg-card__body--plot",
                                ),
                            ],
                            className="pg-card pg-card--plot",
                        ),
                        width=12,
                    ),
                ],
                className="mt-3",
                align="stretch",
            ),
        ],
    )
