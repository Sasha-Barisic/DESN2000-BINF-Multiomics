import dash
from dash import dcc, html, Output, Input, State, ctx
import dash_bootstrap_components as dbc
from django_plotly_dash import DjangoDash
import base64
import io
from dash.exceptions import PreventUpdate
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .clean import clean_first, clean_folder_change, clean_pQ_value
from sklearn.decomposition import PCA

# z = [
#     [0.1, 0.3, 0.5, 0.7, 0.9],
#     [1, 0.8, 0.6, 0.4, 0.2],
#     [0.2, 0, 0.5, 0.7, 0.9],
#     [0.9, 0.8, 0.4, 0.2, 0],
#     [0.3, 0.4, 0.5, 0.7, 1],
# ]

# fig = px.imshow(z, text_auto=True, aspect="auto")

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
            id="upload_data",
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
        dcc.Store(id="store", storage_type="session"),
        dcc.Store(id="cleaned_dataset", storage_type="session"),
        html.Div(id="stored_data_output"),
        html.Br(),
        html.Div(id="filtered_dataset"),
    ]
)


app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(
                    dbc.NavLink(
                        "Github",
                        href="https://github.com/Sasha-Barisic/DESN2000-BINF-Multiomics",
                        style={"color": "white"},
                    )
                ),
            ],
            brand="Multiomics Dashboard",
            brand_style={"color": "white"},
            brand_href="#",
            color="#dc143c",
        ),
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
        ),
        html.Br(),
        html.Div(
            id="output_layout",
        ),
    ],
    style={"width": "80%"},
)


# Callback to switch between tabs.
@app.callback(Output("output_layout", "children"), [Input("tabs", "active_tab")])
def switch_tab(tab_chosen):
    if tab_chosen == "home_tab":
        return radio
    elif tab_chosen == "analysis_tab":
        return analysis_layout


# Callback to store the data into dcc store
@app.callback(
    [Output("store", "data"), Output("stored_data_output", "children")],
    Input("upload_data", "contents"),
    State("upload_data", "filename"),
    prevent_initial_call=True,
)
def store_data(contents, filename):
    # If no file provided prevent refrshing ?.
    if contents is None:
        raise PreventUpdate

    # If uploaded file isn't a CSV, print error message
    if not re.search(".csv", str(filename)):
        return [
            None,
            html.H5(
                "Invalid file format. Please upload a CSV file.", style={"color": "red"}
            ),
        ]

    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    return [
        df.to_json(date_format="iso", orient="split"),
        [
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.H5(
                                        f"The uploaded spreadsheet is: {filename}.",
                                    ),
                                ],
                                width={"size": 10},
                            ),
                            dbc.Col(
                                [
                                    dbc.Button(
                                        "Filter data",
                                        id="filter_button",
                                        style={"background-color": "#DC143C"},
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ],
        # html.H5(f"The uploaded spreadsheet is: {filename}."),
        # dbc.Button(
        #     "Filter data",
        #     id="filter_button",
        #     n_clicks=0,
        #     style={"background-color": "#DC143C"},
        # ),
    ]


import plotly.express as px
import plotly.graph_objects as go


# Callback to filter the data.
@app.callback(
    [Output("cleaned_dataset", "data"), Output("filtered_dataset", "children")],
    Input("filter_button", "n_clicks"),
    State("store", "data"),
)
def filter_dataset(clicks, stored_data):
    # This check always works when callback is fired twice, n_clicks is reset to None after uploading another sheet
    if clicks is not None:
        # Read in the data stored.
        df = pd.read_json(stored_data, orient="split")

        # Filter the dataset.
        separate_cols = clean_first(df)
        clean_cols = clean_folder_change(separate_cols)
        clean_pqvals = clean_pQ_value(clean_cols, True)

        # Create heatmap for whole dataset.
        cols = list(clean_pqvals.columns).remove("unique_id")
        plot_df = pd.melt(clean_pqvals, id_vars=["unique_id"], value_vars=cols)

        # heatmap_fig = go.Figure(
        #     data=go.Heatmap(
        #         x=plot_df["variable"],
        #         y=plot_df["unique_id"],
        #         z=plot_df["value"],
        #         type="heatmap",
        #         colorscale="Viridis",
        #     ),
        # )
        # heatmap_fig.layout.height = 700
        # heatmap_fig.layout.width = 1200
        # heatmap_fig.update_yaxes(tickangle=45, tickfont=dict(color="crimson", size=12))

        # # Create PCA plot for whole dataset.
        # pca_df = clean_pqvals.drop(columns=["unique_id"])
        # pca = PCA(n_components=2)
        # pca_result = pca.fit_transform(pca_df)

        # pca_fig = px.scatter(pca_result[:, 0], pca_result[:, 1])

        return [
            clean_pqvals.to_json(date_format="iso", orient="split"),
            [
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Download filtered dataset",
                                            id="download_csv_button",
                                        ),
                                        dcc.Download(id="download_filtered_data_csv"),
                                    ],
                                    width={"offset": 9},
                                )
                            ]
                        ),
                        # dbc.Row(
                        #     [
                        #         dbc.Col(
                        #             [
                        #                 dcc.Graph(figure=heatmap_fig),
                        #             ]
                        #         ),
                        #     ]
                        # ),
                        # dbc.Row(
                        #     [
                        #         dbc.Col(
                        #             [
                        #                 dcc.Graph(figure=pca_fig),
                        #             ]
                        #         ),
                        #     ]
                        # ),
                    ],
                    # style={
                    #     "border": "1px solid black",
                    #     "background": "white",
                    #     # "border-bottom": "2px solid black",
                    # },
                )
            ],
        ]
    else:
        return [[], []]
    # [html.Br(), dcc.Graph(figure=fig)]


# Download filtered dataset on button click.
@app.callback(
    Output("download_filtered_data_csv", "data"),
    [
        Input("download_csv_button", "n_clicks"),
        State("cleaned_dataset", "data"),
        State("upload_data", "filename"),
    ],
    prevent_initial_call=True,
)
def func(n_clicks, dataset, filename):
    df = pd.read_json(dataset, orient="split")

    filtered_filename = filename.split(".")[0] + "_filtered.csv"
    return dcc.send_data_frame(df.to_csv, filtered_filename, index=False)
