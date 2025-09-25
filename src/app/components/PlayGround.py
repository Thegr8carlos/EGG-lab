import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go

def get_playGround(title: str, description: str, metadata: dict, custom_metadata: dict):
    """
    Layout en 2 filas:
      - Fila 1: Metadata (izq) + Custom Metadata (der)
      - Fila 2: Plot

    Args:
      metadata: dict { clase -> color_css }
      custom_metadata: dict con keys:
        dataset_name, num_classes, sfreq, n_channels, eeg_unit
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
        # orden estable por nombre
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

    # ====== figura demo (placeholder) ======
    fig_demo = go.Figure(
        data=[go.Scatter(
            x=list(range(1, 11)),
            y=[2, 3, 4, 3, 5, 4, 6, 7, 6, 8],
            mode="lines+markers",
            name="Demo"
        )]
    )
    fig_demo.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
                                html.Div("Plot", className="pg-card__title"),
                                html.Div(
                                    dcc.Graph(
                                        id="pg-main-plot",
                                        figure=fig_demo,
                                        responsive=True,
                                        style={"width": "100%", "minHeight": "460px"},
                                        config={"displaylogo": False}
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
