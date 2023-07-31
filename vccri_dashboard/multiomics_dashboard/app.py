import base64
import io
import math
import re

from dash import dcc, html, Output, Input, State, ctx
import dash_bio as dashbio
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from django_plotly_dash import DjangoDash
from functools import partial
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import scipy
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import fdrcorrection

import tempfile

from .clean import clean_first, clean_folder_change, clean_pQ_value

app = DjangoDash(
    "SimpleExample",
    add_bootstrap_links=True,
    external_stylesheets=[
        dbc.themes.MORPH,
    ],
)

#### TABS ######
whole_dataset_analysis_layout = dbc.Container(
    [
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Fulldata Analysis"),
                    ]
                )
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                html.H5("Hierarchical Clustering: Heatmap"),
                dbc.Col(
                    [
                        html.P("Standardization"),
                        dcc.Dropdown(
                            options=[
                                {
                                    "label": "Autoscale features",
                                    "value": "column",
                                },
                                {
                                    "label": "Autoscale samples",
                                    "value": "row",
                                },
                                {"label": "None", "value": "none"},
                            ],
                            value="none",
                            id="global_standardization",
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.P("Distance measure"),
                        dcc.Dropdown(
                            options=[
                                {
                                    "label": "Correlation",
                                    "value": "correlation",
                                },
                                {
                                    "label": "Euclidean",
                                    "value": "euclidean",
                                },
                                {
                                    "label": "Minkowski",
                                    "value": "minkowski",
                                },
                            ],
                            value="euclidean",
                            id="global_distance",
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.P("Clustering method"),
                        dcc.Dropdown(
                            options=[
                                {
                                    "label": "Average",
                                    "value": "average",
                                },
                                {
                                    "label": "Complete",
                                    "value": "complete",
                                },
                                {
                                    "label": "Single",
                                    "value": "single",
                                },
                                {
                                    "label": "Ward",
                                    "value": "ward",
                                },
                            ],
                            value="ward",
                            id="global_clustering",
                        ),
                    ]
                ),
            ],
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            dcc.Graph(id="global_heatmap"),
                        ),
                    ]
                ),
            ]
        ),
        html.Hr(style={"border-top": "2px solid black", "margin": "20px 0"}),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Principle Component Analysis (PCA)"),
                    ]
                )
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P("X-axis PC"),
                        dcc.Dropdown(
                            id="pca_1_dropdown_gl",
                            options=[],
                        ),
                    ],
                    width={"offset": 2, "size": 2},
                ),
                dbc.Col(
                    [
                        html.P("Y-axis PC"),
                        dcc.Dropdown(
                            id="pca_2_dropdown_gl",
                            options=[],
                        ),
                    ],
                    width={"offset": 2, "size": 2},
                ),
            ]
        ),
        html.Br(),
        dbc.Row([html.Div(dcc.Graph(id="pca_plot_gl"))]),
    ],
    id="global_plot_section",
)


pairwise_analysis_layout = (
    dbc.Container(
        [
            html.Div(id="sample_options"),
            html.Br(),
            html.H4("Univariate Analysis"),
            html.Br(),
            dbc.Row(
                [
                    html.H5("Fold Change (FC) Analysis"),
                    html.Br(),
                    dbc.Col(
                        [
                            html.P("Fold change threshold:"),
                            dbc.Input(
                                id="fc_value",
                                type="number",
                                debounce=True,
                                min=2.0,
                                max=5.0,
                                value=2.0,
                            ),
                        ],
                        width={"offset": "1", "size": 2},
                    ),
                ]
            ),
            dcc.Graph(id="fc_plot"),
            html.Br(),
            html.H5("Volcano Plot"),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.RadioItems(
                                options=[
                                    {"label": "Raw", "value": "raw"},
                                    {"label": "FDR", "value": "fdr"},
                                ],
                                value="raw",
                                id="volcano_radioitems",
                                inline=True,
                            ),
                        ],
                        width={"offset": 5},
                    )
                ],
            ),
            html.Div(dcc.Graph(id="volcano_plot")),
            html.H4("Cluster Analysis"),
            html.H5("Hierarchical Clustering: Heatmap"),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P("Standardization"),
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "Autoscale features",
                                        "value": "column",
                                    },
                                    {
                                        "label": "Autoscale samples",
                                        "value": "row",
                                    },
                                    {"label": "None", "value": "none"},
                                ],
                                value="none",
                                id="standardization",
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            html.P("Distance measure"),
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "Correlation",
                                        "value": "correlation",
                                    },
                                    {
                                        "label": "Euclidean",
                                        "value": "euclidean",
                                    },
                                    {
                                        "label": "Minkowski",
                                        "value": "minkowski",
                                    },
                                ],
                                value="euclidean",
                                id="distance",
                            ),
                        ]
                    ),
                    dbc.Col(
                        [
                            html.P("Clustering method"),
                            dcc.Dropdown(
                                options=[
                                    {
                                        "label": "Average",
                                        "value": "average",
                                    },
                                    {
                                        "label": "Complete",
                                        "value": "complete",
                                    },
                                    {
                                        "label": "Single",
                                        "value": "single",
                                    },
                                    {
                                        "label": "Ward",
                                        "value": "ward",
                                    },
                                ],
                                value="ward",
                                id="clustering",
                            ),
                        ]
                    ),
                ]
            ),
            html.Br(),
            html.Div(dcc.Graph(id="heatmap_plot")),
            html.Br(),
            dbc.Row(
                [
                    html.H5("Partitional Clustering: K-Means"),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P("Cluster number (max = 6)"),
                            dbc.Input(
                                id="cluster_value",
                                type="number",
                                min=0,
                                max=6,
                                step=1,
                                value=2,
                            ),
                        ],
                        width={"offset": 2, "size": 2},
                    ),
                ]
            ),
            html.Br(),
            html.Div(dcc.Graph(id="k_means_plot")),
            html.Br(),
            dcc.Store(id="pca_result_store"),
            dbc.Row(
                [
                    html.Hr(
                        style={
                            "border": "1px solid #000000",
                        }
                    ),
                    html.H4("Chemometrics Analysis"),
                    html.H5("Principle Component Analysis (PCA)"),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P("X-axis PC"),
                            dcc.Dropdown(
                                id="pca_1_dropdown",
                                options=[],
                            ),
                        ],
                        width={"offset": 2, "size": 2},
                    ),
                    dbc.Col(
                        [
                            html.P("Y-axis PC"),
                            dcc.Dropdown(
                                id="pca_2_dropdown",
                                options=[],
                            ),
                        ],
                        width={"offset": 2, "size": 2},
                    ),
                ]
            ),
            html.Br(),
            dbc.Row([html.Div(dcc.Graph(id="pca_plot"))]),
            html.Br(),
            dbc.Row(
                [
                    html.Hr(
                        style={
                            "border": "1px solid #000000",
                        }
                    ),
                    html.H4("Classification & Feature Selection"),
                    html.H5("Random Forest"),
                ]
            ),
            html.Div(id="forest_plot"),
        ],
        id="sample_vs_sample_plots",
    ),
)

### ANALYSIS TAB LAYOUT ###
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
        html.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(id="spreadsheet_name"),
                            ]
                        )
                    ]
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button(
                                            "Filter data",
                                            id="filter_button",
                                            style={
                                                "background-color": "#D70040",
                                                "border": "2px solid #042749",
                                            },
                                        ),
                                        dbc.Button(
                                            "Download filtered dataset",
                                            id="download_csv_button",
                                            disabled=True,
                                            style={
                                                "background-color": "#A9A9A9",
                                                "border": "2px solid #042749",
                                            },
                                        ),
                                        dbc.Button(
                                            "Download Plots",
                                            id="download_plots_button",
                                            disabled=True,
                                            style={
                                                "background-color": "#A9A9A9",
                                                "border": "2px solid #042749",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                            width={"offset": 3},
                        ),
                    ],
                ),
                dcc.Download(id="download_filtered_data_csv"),
                dcc.Download(id="download_plots_data"),
                dcc.Loading(html.Div(id="placeholder_loading"), id=""),
            ],
            id="global_buttons",
            style={"visibility": "hidden"},
        ),
        html.Br(),
        html.Br(),
        dbc.Container(
            [
                dbc.Tabs(
                    [
                        dbc.Tab(
                            whole_dataset_analysis_layout,
                            label="Whole dataset analysis",
                            tab_id="whole_tab",
                            active_label_style={"color": "#DA2310"},
                            label_style={"color": "#070707"},
                        ),
                        dbc.Tab(
                            pairwise_analysis_layout,
                            label="Pairwise analysis",
                            tab_id="pairwise_tab",
                            active_label_style={"color": "#DA2310"},
                            label_style={"color": "#070707"},
                        ),
                    ],
                    id="analysis_tabs",
                    active_tab="whole_tab",
                    style={
                        "background-color": "#D8D8D8",
                    },
                ),
            ],
            id="dashboard_tabs",
            style={"visibility": "hidden"},
        ),
        html.Br(),
        html.Div(
            id="output_layout",
        ),
    ],
)

### APP LAYOUT ###
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
                    analysis_layout,
                    label="Analysis",
                    tab_id="analysis_tab",
                    active_label_style={"color": "#DA2310"},
                    label_style={"color": "#070707"},
                ),
            ],
            id="tabs",
            style={  # "border-bottom": "2px solid black",
                "background-color": "#D8D8D8",
            },
        ),
    ],
    style={"width": "80%"},
)

### UPLOADING AND STORING DATA ###


# Callback to store the data into dcc store
@app.callback(
    [
        Output("store", "data"),
        Output("global_buttons", "style"),
        Output("spreadsheet_name", "children"),
    ],
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
        {},
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H6(
                                    f"The uploaded spreadsheet is: {filename}.",
                                ),
                            ],
                            width={"offset": 3},
                        ),
                    ]
                )
            ]
        ),
    ]


# Callback to switch between tabs.
@app.callback(
    Output("output_layout", "children"),
    Input("graph_tabs", "active_tab"),
)
def switch_tab(tab_chosen):
    if tab_chosen == "whole_tab":
        return whole_dataset_analysis_layout
    elif tab_chosen == "pairwise_tab":
        return pairwise_analysis_layout


### PCA FOR WHOLE DATASET ###


# Callback to dynamically generate labels for pca dropwdown.
@app.callback(
    [
        Output("pca_1_dropdown_gl", "value"),
        Output("pca_1_dropdown_gl", "options"),
        Output("pca_2_dropdown_gl", "value"),
        Output("pca_2_dropdown_gl", "options"),
    ],
    [Input("filter_button", "n_clicks")],
)
def populate_global_pca_dropdown(_):
    # Create options for the dropdown menu
    labels = [{"label": lbl + 1, "value": lbl} for lbl in range(0, 8)]

    return labels[0]["value"], labels, labels[1]["value"], labels


# Callback to filter the data, generate heatmap and PCA for whole dataset.
@app.callback(
    [
        Output("cleaned_dataset", "data"),
        Output("sample_options", "children"),
        Output("global_heatmap", "figure"),
        Output("pca_plot_gl", "figure"),
        Output("dashboard_tabs", "style"),
        Output("download_csv_button", "disabled"),
        Output("download_plots_button", "disabled"),
        Output("download_csv_button", "style"),
        Output("download_plots_button", "style"),
    ],
    [
        Input("filter_button", "n_clicks"),
        Input("global_standardization", "value"),
        Input("global_distance", "value"),
        Input("global_clustering", "value"),
        Input("pca_1_dropdown_gl", "value"),
        Input("pca_2_dropdown_gl", "value"),
    ],
    State("store", "data"),
)
def filter_dataset(clicks, gl_std, gl_dist, gl_cl, pca_1_gl, pca_2_gl, stored_data):
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
            id="second_sample", options=labels, value=labels[1]["value"]
        )

        # Heatmap for whole dataset.
        heatmap_df = clean_pqvals.iloc[1:].set_index("unique_id").astype("float")

        gl_heatmap_array = heatmap_df.to_numpy()
        gl_rows = [x.split("-")[0] for x in list(heatmap_df.index)]
        gl_columns = list(heatmap_df.columns.values)

        gl_heatmap_fig = dashbio.Clustergram(
            data=gl_heatmap_array,
            row_labels=gl_rows,
            column_labels=gl_columns,
            standardize=gl_std,
            dist_fun=partial(pdist, metric=gl_dist),
            link_method=gl_cl,
            color_threshold={"row": 250, "col": 700},
            height=800,
            width=1100,
            color_map=[
                [0.0, "#636EFA"],
                [0.25, "#AB63FA"],
                [0.5, "#FFFFFF"],
                [0.75, "#E763FA"],
                [1.0, "#EF553B"],
            ],
        )

        gl_heatmap_fig.update_layout(
            title="Hierarchical Clustering Heatmap for the whole dataset", title_x=0.5
        )
        # Create PCA plot for whole dataset.
        # pylint: disable=no-member
        pca_gl_df = clean_pqvals.set_index("unique_id").T

        # Get sample labels and length of unique lbls.
        sample_labels_gl = list(pca_gl_df.label)
        unique_label_len = len(list(set(list(pca_gl_df.label))))

        # Drop column label
        pca_gl_df = pca_gl_df.drop(columns="label")

        # PCA
        pca = PCA(n_components=unique_label_len)
        pca_result_gl = pca.fit_transform(pca_gl_df)
        pca_result_df_gl = pd.DataFrame(pca_result_gl)

        # Add labels to df
        pca_result_df_gl["labels"] = sample_labels_gl

        # Plot figure.
        pca_fig_gl = px.scatter(
            pca_result_df_gl,
            x=pca_1_gl,
            y=pca_2_gl,
            color=pca_result_df_gl["labels"],
            title="Principal Component Analaysis (PCA) of the whole dataset",
        )
        pca_fig_gl.update_layout(legend_title_text="Sample")
        pca_fig_gl.update_layout(
            xaxis_title=f"PC{int(pca_1_gl) + 1}", yaxis_title=f"PC{int(pca_2_gl) + 1}"
        )
        pca_fig_gl.update_layout(title_x=0.5)
        return [
            clean_pqvals.to_json(date_format="iso", orient="split"),
            [
                dbc.Container(
                    [
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
                    ],
                ),
            ],
            gl_heatmap_fig,
            pca_fig_gl,
            {},
            False,
            False,
            {
                "background-color": "#D70040",
                "border": "2px solid #042749",
            },
            {
                "background-color": "#D70040",
                "border": "2px solid #042749",
            },
        ]

    else:
        return [
            [],
            [],
            [],
            [],
            {"visibility": "hidden"},
            True,
            True,
            {
                "background-color": "#A9A9A9",
                "border": "2px solid #042749",
            },
            {
                "background-color": "#A9A9A9",
                "border": "2px solid #042749",
            },
        ]


### DOWNLOAD FILTERED DATA ###
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


### SAMPLE VS SAMPLE PLOTS ###


# Create fc plot based on selected dropdown feature
@app.callback(
    Output("fc_plot", "figure"),
    [
        Input("first_sample", "value"),
        Input("second_sample", "value"),
        Input("fc_value", "value"),
    ],
    State("cleaned_dataset", "data"),
)
def fc_plot(first_sample, second_sample, fc_val, dataset):
    # Extract from the data the two samples
    df = pd.read_json(dataset, orient="split")
    # pylint: disable=no-member
    df = df.iloc[1:]

    first_df = df.filter(regex=f"{first_sample}").astype(float)
    second_df = df.filter(regex=f"{second_sample}").astype(float)
    unique_id_series = df.unique_id

    # Calculate the fold change and set the threshold vlas
    fold_change = first_df.mean(axis=1) / second_df.mean(axis=1)
    lower_significance_threshold = 1 / fc_val
    upper_significance_threshold = fc_val

    fc_df = pd.DataFrame({"Fold Change": fold_change})

    sigs = []
    for _, row in fc_df.iterrows():
        if row["Fold Change"] <= lower_significance_threshold:
            sigs.append("Sig.Down")
        elif row["Fold Change"] >= upper_significance_threshold:
            sigs.append("Sig.Up")
        else:
            sigs.append("Unsig.")

    fc_df["Significance"] = sigs
    fc_df["unique_id"] = unique_id_series
    fc_df["unique_id"] = fc_df["unique_id"].apply(lambda x: x.split("-")[0])
    fc_df["Fold Change"] = np.log2(fc_df["Fold Change"])

    fc_fig = px.scatter(
        fc_df,
        x="unique_id",
        y="Fold Change",
        color="Significance",
        hover_data=["unique_id"],
        labels={"Fold Change": "log2(FC)", "unique_id": "Unique ID"},
        title=f"Fold change analysis: {first_sample} vs {second_sample}",
    )
    fc_fig.update_layout(title_x=0.5)

    return fc_fig


# Create volcano plot based on selected dropdown feature
@app.callback(
    Output("volcano_plot", "figure"),
    [
        Input("first_sample", "value"),
        Input("second_sample", "value"),
        Input("volcano_radioitems", "value"),
    ],
    State("cleaned_dataset", "data"),
)
def volcano_plot(first_sample, second_sample, p_option, dataset):
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

    # If FDR is selected
    if p_option == "fdr":
        p_values = fdrcorrection(p_values, alpha=0.05, method="indep", is_sorted=False)[
            1
        ]

    ### HEREEEEEEEEEEEEE
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
        title=f"Volcano plot: {first_sample} vs {second_sample}",
    )
    volcano_fig.add_vline(x=-1, line_width=2, line_dash="dash", line_color="black")
    volcano_fig.add_vline(x=1, line_width=2, line_dash="dash", line_color="black")
    volcano_fig.add_hline(
        y=-math.log10(0.05), line_width=2, line_dash="dash", line_color="black"
    )
    volcano_fig.update_layout(title_x=0.5)

    return volcano_fig


# Create heatmap based on selected dropdown feature.
@app.callback(
    Output("heatmap_plot", "figure"),
    [
        Input("first_sample", "value"),
        Input("second_sample", "value"),
        Input("standardization", "value"),
        Input("distance", "value"),
        Input("clustering", "value"),
    ],
    State("cleaned_dataset", "data"),
)
def heatmap(first_sample, second_sample, stand, distance, cluster, dataset):
    # Extract from the data the two samples
    df = pd.read_json(dataset, orient="split")
    # pylint: disable=no-member
    df = df.iloc[1:]

    two_samples_df = (
        df.filter(regex=f"unique_id|{first_sample}|{second_sample}")
        .set_index("unique_id")
        .astype("float")
    )
    heatmap_array = two_samples_df.to_numpy()
    rows = [x.split("-")[0] for x in list(two_samples_df.index)]
    columns = list(two_samples_df.columns.values)

    heatmap_fig = dashbio.Clustergram(
        data=heatmap_array,
        row_labels=rows,
        column_labels=columns,
        standardize=stand,
        dist_fun=partial(pdist, metric=distance),
        link_method=cluster,
        color_threshold={"row": 250, "col": 700},
        height=800,
        width=1100,
    )

    heatmap_fig.update_layout(
        title=f"Hierarchical Clustering Heatmap: {first_sample} vs {second_sample}",
        title_x=0.5,
    )
    return heatmap_fig


@app.callback(
    [
        Output("pca_1_dropdown", "value"),
        Output("pca_1_dropdown", "options"),
        Output("pca_2_dropdown", "value"),
        Output("pca_2_dropdown", "options"),
    ],
    [Input("filter_button", "n_clicks")],
)
def populate_pca_dropdown(_):
    # Create options for the dropdown menu
    labels = [{"label": lbl + 1, "value": lbl} for lbl in range(0, 4)]

    return labels[0]["value"], labels, labels[1]["value"], labels


# Create PCA plot
@app.callback(
    [
        Output("pca_result_store", "data"),
        Output("pca_plot", "figure"),
        # Output("sample_vs_sample_pca_dropdown", "style"),
    ],
    [
        Input("first_sample", "value"),
        Input("second_sample", "value"),
        Input("pca_1_dropdown", "value"),
        Input("pca_2_dropdown", "value"),
    ],
    [
        State("cleaned_dataset", "data"),
    ],
)
def pca(first_sample, second_sample, pca_1, pca_2, dataset):
    # Extract from the data the two samples
    df = pd.read_json(dataset, orient="split")

    # pylint: disable=no-member
    df.set_index("unique_id", inplace=True)
    df = df.filter(regex=f"{first_sample}|{second_sample}").T

    # Get sample labels and length of unique lbls.
    sample_labels = list(df.label)
    # unique_label_len = len(list(set(list(df.label))))

    # Drop column
    df = df.drop(columns="label")

    # PCA
    pca = PCA(n_components=6)
    pca_result = pca.fit_transform(df)
    pca_result_df = pd.DataFrame(pca_result)

    # Add labels to df
    pca_result_df["labels"] = sample_labels

    # Plot figure.
    pca_fig = px.scatter(
        pca_result_df,
        x=pca_1,
        y=pca_2,
        color=pca_result_df["labels"],
        title=f"Principle Component Analysis (PCA): {first_sample} vs {second_sample}",
    )
    pca_fig.update_layout(legend_title_text="Sample")
    pca_fig.update_layout(
        xaxis_title=f"PC{int(pca_1) + 1}", yaxis_title=f"PC{int(pca_2) + 1}"
    )
    pca_fig.update_layout(title_x=0.5)

    return [
        pca_result_df.to_json(date_format="iso", orient="split"),
        pca_fig,
    ]


# Create a K-means plot.
@app.callback(
    Output("k_means_plot", "figure"),
    [
        Input("cluster_value", "value"),
        Input("pca_result_store", "data"),
    ],
    [
        State("cleaned_dataset", "data"),
    ],
)
def k_means(n_cluster, dataset, clean):
    # Extract from the data the two samples
    # pylint: disable=no-member
    df = pd.read_json(dataset, orient="split")

    df.drop(columns="labels", inplace=True)
    # Initialize the class object
    kmeans = KMeans(n_clusters=n_cluster)

    # predict the labels of clusters.
    label = kmeans.fit_predict(df)
    unique_label = np.unique(kmeans.fit_predict(df))

    traces = []
    for i, u_lbl in enumerate(unique_label):
        filtered_labels = df[label == u_lbl]

        traces.append(
            go.Scatter(
                x=filtered_labels[0],
                y=filtered_labels[1],
                mode="markers",
                name=f"Cluster {i}",
            )
        )

    kmeans_fig = make_subplots(specs=[[{"secondary_y": True}]])

    for tr in traces:
        kmeans_fig.add_trace(tr)

    kmeans_fig.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend_title="Clusters",
        title=f"K-means: {n_cluster} clusters",
    )
    kmeans_fig.update_layout(title_x=0.5)

    return kmeans_fig


# # Create a Random Forest VIP plot.
@app.callback(
    Output("forest_plot", "figure"),
    [
        Input("first_sample", "value"),
        Input("second_sample", "value"),
        Input("cleaned_dataset", "data"),
    ],
)
def random_vip(first_sample, second_sample, dataset):
    # Read in data and filter for the two samples.
    # pylint: disable=no-member
    df = pd.read_json(dataset, orient="split")
    df = df.set_index(
        "unique_id",
    )

    df = df.filter(regex=f"{first_sample}|{second_sample}").T
    sample_labels = list(df.label)
    unique_ids = list(df.columns)
    unique_ids.remove("label")

    df.drop(columns=["label"], inplace=True)

    features = df

    # Encode labels
    encoded_labels = LabelEncoder()
    enc_labels = encoded_labels.fit_transform(sample_labels)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, enc_labels, test_size=0.2, random_state=42
    )
    # Random Forest Classifier
    forest = RandomForestClassifier(n_estimators=500)
    forest.fit(X_train, y_train)
    results = forest.predict_proba(X_test)

    # Feature Importances
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = features.columns

    new_labels = []
    for i in range(len(unique_ids)):
        new_labels.append(unique_ids[indices[i]].split("-")[0])

    forest_fig = px.scatter(y=new_labels, x=np.flip(importances[indices]))

    return forest_fig


### DOWNLOAD PLOTS ###
@app.callback(
    [
        Output("download_plots_data", "data"),
        Output("placeholder_loading", "children"),
    ],
    [
        Input("download_plots_button", "n_clicks"),
    ],
    [
        State("global_heatmap", "figure"),
        State("pca_plot_gl", "figure"),
        State("volcano_plot", "figure"),
        State("heatmap_plot", "figure"),
        State("k_means_plot", "figure"),
        State("pca_plot", "figure"),
        State("forest_plot", "figure"),
    ],
    prevent_initial_call=True,
)
def download_plot_to_pdf(
    _,
    gl_heatmap,
    pca_plot_gl,
    volcano_plot,
    heatmap_plot,
    k_means_plot,
    pca_plot,
    forest_plot,
):
    """Function to download the main plot figure to a PDF file to the client's
        machine

    Args:
        _ (dbc.Button.n_clicks): the "Download Plot" button nclicks attribute that
        triggers the callback
        fig (plotly.graph_objects.Figure): Plotly figure that will be downloaded

    Returns:
        dcc.Download.data : the plot that will be sent ot the Download component
        for eventual downloading
    """

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
        temp_fig = go.Figure(gl_heatmap)
        pio.write_image(temp_fig, temp_file.name, height=780, width=1200)
        return dcc.send_file(temp_file.name), ""
