import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from django_plotly_dash import DjangoDash

app = DjangoDash(
    "SimpleExample",
    add_bootstrap_links=True,
    external_stylesheets=[
        dbc.themes.MINTY,
    ],
)

app.layout = html.Div(
    [
        dcc.RadioItems(
            id="dropdown-color",
            options=[
                {"label": c, "value": c.lower()} for c in ["Red", "Green", "Blue"]
            ],
            value="red",
        ),
        html.Div(id="output-color"),
        dcc.RadioItems(
            id="dropdown-size",
            options=[
                {"label": i, "value": j}
                for i, j in [("L", "large"), ("M", "medium"), ("S", "small")]
            ],
            value="medium",
        ),
        dbc.Button("Primary", color="primary", className="me-1"),
        dbc.Button("Secondary", color="secondary", className="me-1"),
    ]
)


@app.callback(
    dash.dependencies.Output("output-color", "children"),
    [dash.dependencies.Input("dropdown-color", "value")],
)
def callback_color(dropdown_value):
    return "The selected color is %s." % dropdown_value
