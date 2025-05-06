import dash_bootstrap_components as dbc 
from dash import html


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
                "Filtros",
                href="/filtros",
                active="exact",
                className="nav-link"
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Extractores de Caracteristicas",
                href="/extractores",
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
                "Modelado Habla Interna",
                href="/hablainterna",
                active="exact",
                className="nav-link"
            )
        )
    ]

    return dbc.Navbar(
        dbc.Container([
            # 1) left menu nav brand w
            dbc.NavbarBrand("Inicio", href="/", className="nav-link"),
            # 2) nav link with all the pages
            dbc.Nav(nav_links, navbar=True, fill=True, justified="center", className="mx-auto"),
        ], fluid=True),
        color="dark",
        dark=True,
        sticky="top",
        expand="lg",
        className="navbar-custom"
    )




