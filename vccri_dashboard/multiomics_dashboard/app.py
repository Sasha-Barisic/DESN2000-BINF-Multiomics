import base64
import io
import math
import re


from dash import dcc, html, Output, Input, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from django_plotly_dash import DjangoDash
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.decomposition import PCA


from .clean import clean_first, clean_folder_change, clean_pQ_value

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
        dcc.Loading(
            html.Div(id="filtered_dataset"),
            type="cube",
            color="red",
        ),
        html.Br(),
        html.Div(id="sample_vs_sample_volcano"),
        html.Br(),
        html.Div(id="sample_vs_sample_heatmap"),
        html.Br(),
        dcc.Store(id="pca_result_store"),
        dbc.Container(id="sample_vs_sample_pca_dropdown"),
        html.Br(),
        html.Div(id="sample_vs_sample_pca_res"),
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

    _, content_string = contents.split(",")
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


# Callback to filter the data.
@app.callback(
    [Output("cleaned_dataset", "data"), Output("filtered_dataset", "children")],
    Input("filter_button", "n_clicks"),
    State("store", "data"),
)
def filter_dataset(clicks, stored_data):
    # This check always works when callback is fired twice,
    # n_clicks is reset to None after uploading another sheet
    if clicks is not None:
        # Read in the data stored.
        df = pd.read_json(stored_data, orient="split")

        # Filter the dataset.
        separate_cols = clean_first(df)
        clean_cols = clean_folder_change(separate_cols)
        clean_pqvals = clean_pQ_value(clean_cols, True)

        # Create options for the dropdown menu
        samples = [x.split(".")[0] for x in list(clean_pqvals.columns)]
        labels = [
            {"label": lbl, "value": lbl}
            for lbl in sorted(list(set(samples)))
            if not re.search("lank|unique_id", lbl)
        ]

        dropdown_menu_1 = dcc.Dropdown(
            id="first_sample", options=labels, value=labels[0]["value"]
        )
        dropdown_menu_2 = dcc.Dropdown(
            id="second_sample", options=labels, value=labels[4]["value"]
        )

        # Create heatmap for whole dataset.
        # cols = list(clean_pqvals.columns).remove("unique_id")
        # plot_df = pd.melt(clean_pqvals, id_vars=["unique_id"], value_vars=cols)

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
                html.Hr(
                    style={
                        "border": "1px solid #000000",
                    }
                ),
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
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dropdown_menu_1,
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dropdown_menu_2,
                                    ]
                                ),
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
                ),
            ],
        ]
    else:
        return [[], []]


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
def func(_, dataset, filename):
    df = pd.read_json(dataset, orient="split")

    filtered_filename = filename.split(".")[0] + "_filtered.csv"
    # pylint: disable=no-member
    return dcc.send_data_frame(df.to_csv, filtered_filename, index=False)


# Create volcano plot based on selected dropdown feature
@app.callback(
    Output("sample_vs_sample_volcano", "children"),
    [Input("first_sample", "value"), Input("second_sample", "value")],
    State("cleaned_dataset", "data"),
)
def volcano_plot(first_sample, second_sample, dataset):
    # Extract from the data the two samples
    df = pd.read_json(dataset, orient="split")
    # pylint: disable=no-member
    df = df.iloc[1:]

    first_df = df.filter(regex=f"{first_sample}").astype(float)
    second_df = df.filter(regex=f"{second_sample}").astype(float)
    unique_id_series = df.unique_id

    # p-values
    p_values = []
    for i in range(len(first_df)):
        first_values = first_df.iloc[i, :].values
        first_values = [float(x) for x in first_values]
        second_values = second_df.iloc[i, :].values
        second_values = [float(x) for x in second_values]
        _, p_value = stats.ttest_ind(first_values, second_values)
        p_values.append(p_value)

    fold_change = np.log2(first_df.mean(axis=1) / second_df.mean(axis=1))
    significance_threshold = 0.05

    volcano_df = pd.DataFrame({"Fold Change": fold_change, "p-value": p_values})

    sigs = []
    for _, row in volcano_df.iterrows():
        if row["p-value"] < significance_threshold and (
            row["Fold Change"] <= -1 or row["Fold Change"] >= 1
        ):
            sigs.append(True)
        else:
            sigs.append(False)
    volcano_df["Significant"] = sigs
    volcano_df["unique_id"] = unique_id_series

    volcano_df["p-value"] = -np.log10(volcano_df["p-value"])

    volcano_fig = px.scatter(
        volcano_df,
        x="Fold Change",
        y="p-value",
        color="Significant",
        hover_data=["unique_id"],
        labels={"Fold Change": "log2(FC)", "p-value": "-log10(p)"},
    )
    volcano_fig.add_vline(x=-1, line_width=2, line_dash="dash", line_color="black")
    volcano_fig.add_vline(x=1, line_width=2, line_dash="dash", line_color="black")
    volcano_fig.add_hline(
        y=-math.log10(0.05), line_width=2, line_dash="dash", line_color="black"
    )

    return [
        html.Br(),
        html.H4("Univariate Analysis"),
        dcc.Graph(figure=volcano_fig),
    ]


# Create heatmap based on selected dropdown feature
@app.callback(
    Output("sample_vs_sample_heatmap", "children"),
    [Input("first_sample", "value"), Input("second_sample", "value")],
    State("cleaned_dataset", "data"),
)
def heatmap(first_sample, second_sample, dataset):
    # Extract from the data the two samples
    df = pd.read_json(dataset, orient="split")
    # pylint: disable=no-member
    df = df.iloc[1:]
    unique_id_series = df.unique_id

    two_samples_df = df.filter(regex=f"{first_sample}|{second_sample}")
    heatmap_fig = px.imshow(
        two_samples_df,
        x=two_samples_df.columns,
        y=unique_id_series,
        color_continuous_scale="RdBu",
    )
    heatmap_fig.update_layout(width=900, height=500)
    return [
        html.Br(),
        dcc.Graph(figure=heatmap_fig),
    ]


# Create PCA plot
@app.callback(
    [
        Output("sample_vs_sample_pca_dropdown", "children"),
        Output("pca_result_store", "data"),
    ],
    [Input("first_sample", "value"), Input("second_sample", "value")],
    [
        State("cleaned_dataset", "data"),
    ],
)
def pca(first_sample, second_sample, dataset):
    # Extract from the data the two samples
    df = pd.read_json(dataset, orient="split")

    # pylint: disable=no-member
    df.set_index("unique_id", inplace=True)
    df = df.filter(regex=f"{first_sample}|{second_sample}").T

    # Get sample labels
    sample_labels = list(df.label)

    # Drop column
    df = df.drop(columns="label")

    # PCA
    pca = PCA(n_components=6)
    pca_result = pca.fit_transform(df)
    pca_result_df = pd.DataFrame(pca_result)

    # Construct the labels for the dropdown menus

    pca_len = len(pca_result_df.columns)
    labels = [
        {"label": lbl + 1, "value": f"first_pca_{lbl}"} for lbl in range(0, pca_len - 2)
    ]
    pca_drop_1 = dcc.Dropdown(
        id="pca_1_dropdown",
        options=labels,
    )

    pca_drop_2 = dcc.Dropdown(
        id="pca_2_dropdown",
        options=labels,
    )

    pca_result_df["labels"] = sample_labels

    pca_fig = px.scatter(pca_result_df, x=0, y=1, color=pca_result_df["labels"])
    pca_fig.update_layout(legend_title_text="Sample")
    return [
        [
            html.Hr(
                style={
                    "border": "1px solid #000000",
                },
            ),
            html.Br(),
            html.H4("Chemometrics Analysis"),
            html.Br(),
            dbc.Container(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.P("X-axis PC"),
                                    pca_drop_1,
                                ],
                                width={"offset": 2, "size": 2},
                            ),
                            dbc.Col(
                                [
                                    html.P("Y-axis PC"),
                                    pca_drop_2,
                                ],
                                width={"offset": 2, "size": 2},
                            ),
                        ]
                    )
                ],
            ),
        ],
        pca_result_df.to_json(date_format="iso", orient="split"),
    ]


# filter button triggers dropdown generation
# the dropdown triggers plot generation
@app.callback(
    Output("sample_vs_sample_pca_res", "children"),
    [
        Input("pca_1_dropdown", "value"),
        Input("pca_2_dropdown", "value"),
        Input("pca_result_store", "data"),
    ],
)
def update_pca_graph(pca_1, pca_2, pca_data):
    # Extract PCA data
    df = pd.read_json(pca_data, orient="split")

    # Check if values are selected
    if pca_1 is not None and pca_2 is not None:
        first_sample = int(pca_1.split("_")[-1])
        second_sample = int(pca_2.split("_")[-1])

        pca_fig = px.scatter(
            df,
            x=first_sample,
            y=second_sample,
            color=df["labels"],
            # labels={
            #     first_sample: f"PC{first_sample}",
            #     second_sample: f"PC{second_sample}",
            # },
        )
        pca_fig.update_layout(legend_title_text="Sample")

        return [dcc.Graph(figure=pca_fig)]
    else:
        return []
