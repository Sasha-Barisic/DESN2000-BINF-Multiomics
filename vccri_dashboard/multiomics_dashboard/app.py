import dash
from dash import dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
from django_plotly_dash import DjangoDash
import base64
import io
from dash.exceptions import PreventUpdate
import re
import pandas as pd


app = DjangoDash(
    "SimpleExample",
    add_bootstrap_links=True,
    external_stylesheets=[
        dbc.themes.MORPH,
    ],
)


radio = dbc.Container(
    [
        dcc.RadioItems(
            id="dropdown-color",
            options=[
                {"label": c, "value": c.lower()} for c in ["Red", "Green", "Blue"]
            ],
            value="red",
        ),
    ]
)

analysis_layout = dbc.Container(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                [html.A("Click to upload data.", style={"font-size": "140%"})]
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
                "background-color": "#D8D8D8",
            },
        ),
        dcc.Store(id="store"),
        html.Div(id="output-data-upload"),
    ]
)


app.layout = dbc.Container(
    [
        html.H1("PLACEHOLDER FOR HEADER"),
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Home",
                    tab_id="home_tab",
                    active_label_style={"color": "#DA2310"},
                    label_style={"color": "#070707"},
                ),
                dbc.Tab(
                    label="Analysis",
                    tab_id="analysis_tab",
                    active_label_style={"color": "#DA2310"},
                    label_style={"color": "#070707"},
                ),
            ],
            id="tabs",
            active_tab="home_tab",
            style={  # "border-bottom": "2px solid black",
                "background-color": "#D8D8D8",
            },
            persistence_type="session",
        ),
        html.Br(),
        html.Div(
            id="output_layout",
        ),
    ],
    style={"width": "80%"},
)


@app.callback(
    Output("output-data-upload", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
)
def update_output(contents, filename):
    # If no file provided prevent refrshing ?.
    if contents is None:
        raise PreventUpdate

    # If uploaded file isn't a CSV, print error message
    if not re.search(".csv", str(filename)):
        return [
            html.H5(
                "Invalid file format. Please upload a CSV file.", style={"color": "red"}
            )
        ]

    content_type, content_string = contents.split(",")
    print(content_type)
    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    print(df)
    return [
        html.H5(f"The uploaded spreadsheet is: {filename}."),
        dbc.Button("Filter data", color="primary", className="me-1"),
    ]


@app.callback(Output("output_layout", "children"), [Input("tabs", "active_tab")])
def switch_tab(tab_chosen):
    if tab_chosen == "home_tab":
        return radio
    elif tab_chosen == "analysis_tab":
        return analysis_layout


@app.callback(
    dash.dependencies.Output("output-color", "children"),
    [dash.dependencies.Input("dropdown-color", "value")],
)
def callback_color(dropdown_value):
    return "The selected color is %s." % dropdown_value
