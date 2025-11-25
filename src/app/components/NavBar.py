import dash_bootstrap_components as dbc
from dash import html, dcc


def get_navBar(page_registry):
    # nav_links = [
    #     dbc.NavItem(
    #         dbc.NavLink(
    #             page["name"],
    #             href=page["path"],
    #             active="exact",
    #             className="nav-link"
    #         )
    #     )
    #     for page in page_registry.values()
    #     if page["path"] != "/"
    # ]
    nav_links = [
        dbc.NavItem(
            dbc.NavLink(
                "Cargar Datos",
                href="/cargardatos",
                active="exact",
                className="nav-link"
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Dataset",
                href="/dataset",
                active="exact",
                className="nav-link"
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Filtros",
                href="/filtros",
                active="exact",
                className="nav-link"
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Modelado P300",
                href="/p300",
                active="exact",
                className="nav-link"
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Modelado Inner Speech",
                href="/inner-speech",
                active="exact",
                className="nav-link"
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Simulación",
                href="/simulation",
                active="exact",
                className="nav-link"
            )
        )
    ]

    return dbc.Navbar(
        dbc.Container([
            # 1) left menu nav brand w
            dbc.NavbarBrand(
                html.Img(id="gif-btn",src="/assets/media/egg-gif.gif", height="40px"),
            ),
            # 2) nav link with all the pages
            dbc.Nav(nav_links, navbar=True, fill=True, justified="center", className="mx-auto"),
            # 3) right side controls (experiment ID and action buttons)
            html.Div([
                # Experiment ID indicator
                html.Div([
                    html.Span("Experimento: ", style={"color": "#adb5bd", "fontSize": "0.9rem"}),
                    html.Span(id="navbar-experiment-id", children="--", style={"color": "#0dcaf0", "fontWeight": "bold", "fontSize": "0.9rem"})
                ], style={"display": "inline-block", "marginRight": "15px"}),
                # New Experiment button
                dbc.Button(
                    [html.I(className="fas fa-plus-circle me-1"), "Nuevo"],
                    id="btn-new-experiment",
                    color="success",
                    size="sm",
                    className="me-2",
                    title="Crear nuevo experimento"
                ),
                # Clear Cache button
                dbc.Button(
                    [html.I(className="fas fa-trash-alt me-1"), "Limpiar"],
                    id="btn-clear-cache",
                    color="warning",
                    size="sm",
                    title="Limpiar cache de experimentos antiguos"
                ),
            ], style={"display": "flex", "alignItems": "center"}),
        ], fluid=True),
        color="dark",
        dark=True,
        sticky="top",
        expand="lg",
        className="navbar-custom"
    )




