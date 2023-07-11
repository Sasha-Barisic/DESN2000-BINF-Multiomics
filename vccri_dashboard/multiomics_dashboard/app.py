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
        dbc.Container(
            [
                html.H4("Univariate Analysis"),
                html.H5("Volcano Plot"),
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
                html.Div(id="volcano_plot"),
            ],
            id="sample_vs_sample_volcano",
            style={"visibility": "hidden"},
        ),
        html.Br(),
        dbc.Container(
            [
                html.H4("Cluster Analysis"),
                html.H5("Hierarchical Clustering: Heatmap"),
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
                html.Div(id="heatmap_plot"),
            ],
            id="sample_vs_sample_heatmap",
            style={"visibility": "hidden"},
        ),
        html.Br(),
        dbc.Container(
            [
                dbc.Row(
                    [
                        html.H5("Partitional Clustering: K-Means"),
                    ]
                ),
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
                html.Div(id="k_means_plot"),
            ],
            id="sample_vs_sample_k_means",
            style={"visibility": "hidden"},
        ),
        html.Br(),
        dcc.Store(id="pca_result_store"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        html.Hr(
                            style={
                                "border": "1px solid #000000",
                            }
                        ),
                        html.H4("Chemometrics Analysis"),
                    ]
                ),
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
                dbc.Row([html.Div(id="pca_plot")]),
            ],
            id="sample_vs_sample_pca_dropdown",
            style={"visibility": "hidden"},
        ),
        dbc.Container(
            [
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
            id="sample_vs_sample_random",
            style={"visibility": "hidden"},
        ),
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
    [Output("sample_vs_sample_volcano", "style"), Output("volcano_plot", "children")],
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

    return {}, [
        html.Br(),
        dcc.Graph(figure=volcano_fig),
    ]


# Create heatmap based on selected dropdown feature
@app.callback(
    [Output("sample_vs_sample_heatmap", "style"), Output("heatmap_plot", "children")],
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
    return {}, [
        html.Br(),
        dcc.Graph(figure=heatmap_fig),
    ]


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
        Output("pca_plot", "children"),
        Output("sample_vs_sample_pca_dropdown", "style"),
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

    # Get sample labels
    sample_labels = list(df.label)

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
    )
    pca_fig.update_layout(legend_title_text="Sample")
    pca_fig.update_layout(
        xaxis_title=f"PC{int(pca_1) + 1}", yaxis_title=f"PC{int(pca_2) + 1}"
    )

    return [
        pca_result_df.to_json(date_format="iso", orient="split"),
        dcc.Graph(figure=pca_fig),
        {},
    ]


# Create a K-means plot.
@app.callback(
    [
        Output("k_means_plot", "children"),
        Output("sample_vs_sample_k_means", "style"),
    ],
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
        xaxis_title="PC1", yaxis_title="PC2", legend_title="Clusters"
    )

    return [
        dcc.Graph(figure=kmeans_fig),
        {},
    ]


# # Create a Random Forest VIP plot.
# @app.callback(
#     [
#         Output("forest_plot", "children"),
#         Output("sample_vs_sample_random", "style"),
#     ],
#     [
#         Input("first_sample", "value"),
#         Input("second_sample", "value"),
#         Input("cleaned_dataset", "data"),
#     ],
# )
def random_vip(first_sample, second_sample, dataset):
    # Read in data and filter for the two samples.
    # pylint: disable=no-member
    df = pd.read_json(dataset, orient="split")

    df_1 = df.filter(regex=f"unique_id|{first_sample}")
    df_1 = df_1.iloc[1:]
    first_col = list(df_1.columns)

    df_2 = df.filter(regex=f"unique_id|{second_sample}")
    df_2 = df_2.iloc[1:]

    second_col = list(df_2.columns)

    rename_cols = {}
    for i, col in enumerate(first_col):
        rename_cols[second_col[i]] = col

    df_2 = df_2.rename(columns=rename_cols)

    # Extract features (X) and labels (y)
    features = pd.concat([df_1, df_2], axis=0, ignore_index=True).set_index("unique_id")
    labels = [f"{first_sample}"] * len(df_1) + [f"{second_sample}"] * len(df_2)

    # Encode labels
    encoded_labels = LabelEncoder()
    enc_labels = encoded_labels.fit_transform(labels)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, enc_labels, test_size=0.5, shuffle=False
    )
    # Random Forest Classifier
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train, y_train)
    results = forest.predict_proba(X_test)

    print(df.unique_id.unique())
    print(results)
    print(y_test)

    # Feature Importances
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = features.columns

    forest_fig = px.bar(range(features.shape[1]), importances[indices])

    return dcc.Graph(figure=forest_fig), {}
