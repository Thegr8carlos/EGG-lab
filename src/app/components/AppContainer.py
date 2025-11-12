from dash import html, dcc


def get_app_container(header, navBar, sidebar, page_container):
    """Return the main application container layout.

    Params:
        header: header component
        navBar: navigation bar component
        sidebar: sidebar component (already produced by get_sideBar)
        page_container: Dash page_container
    """

    return html.Div(
        id="app-container",
        children=[
            header,
            navBar,
            html.Div(
                id="main-content-wrapper",
                style={"display": "flex", "flexDirection": "row"},
                children=[
                    html.Div(
                        id="sidebar-wrapper",
                        children=sidebar,
                        className="sideBar-container",
                    ),
                    html.Div(
                        id="page-content",
                        children=page_container,
                        style={"flex": 1, "padding": "20px"},
                    ),
                ],
            ),
            dcc.Location(id="url"),
            dcc.Store(id="selected-file-path", storage_type="local"),
            dcc.Store(id="selected-dataset", storage_type="session"),
        ],
    )
