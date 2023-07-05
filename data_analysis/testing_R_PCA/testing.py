import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter


# Import spreadsheet - two samples
# two_samples_df = pd.read_csv("two_samples.csv", sep=",", index_col="unique_id").T

# labels = list(two_samples_df.label)

# two_samples_df = two_samples_df.drop(columns=["label"]).astype("float64")

# print(two_samples_df)
# two__df = pd.read_csv("data_normalized.csv", sep=",")

# # Get the list of labels
# labels = list(two_samples_df.Label)
# print(labels)
# two_samples_df.drop(columns=["Label"], inplace=True)

# # Convert pd.df to r.df
# with localconverter(ro.default_converter + pandas2ri.converter):
#     r_from_pd_df = ro.conversion.py2rpy(two_samples_df)

# # # Do the R PCA method
# ro.globalenv["r_from_pd_df"] = r_from_pd_df
# ro.r(
#     f"""
#     pca <- prcomp(r_from_pd_df, center=TRUE, scale=F);

#     # obtain variance explained
#     sum.pca <- summary(pca);
#     imp.pca <- sum.pca$importance;
#     std.pca <- imp.pca[1,]; # standard devietation
#     var.pca <- imp.pca[2,]; # variance explained by each PC
#     cum.pca <- imp.pca[3,]; # cummulated variance explained

#     # store the item to the pca object
#     pca_result <- append(pca, list(std=std.pca, variance=var.pca, cum.var=cum.pca))

#     pca_table <- pca_result$x
#     print(pca_table)
# """
# )

# Convert the r.df back to pd.df
# pca_table = ro.globalenv["pca_table"]
# with (ro.default_converter + pandas2ri.converter).context():
#     pd_from_r_df = ro.conversion.get_conversion().rpy2py(pca_table)

# pd_from_r_df = pd.DataFrame(pd_from_r_df)

# fig = px.scatter(pd_from_r_df, x=0, y=1, color=labels)
# fig.show()


# pca = PCA(n_components=6)
# pca_result = pca.fit_transform(df)

# pca_result_df = pd.DataFrame(pca_result)
# print(pca_result_df)

df = pd.read_csv("two_samples.csv", sep=",", index_col="unique_id").T.drop(
    columns=["label"]
)


# # cz vs cl1
# db_cz = df.filter(regex="CZ").astype(float)
# db_cl = df.filter(regex="CL").astype(float)
pca = PCA(n_components=6)
pca_result = pca.fit_transform(df)

pca_result_df = pd.DataFrame(pca_result)


# ## PCA
# combined_data = pd.concat([db_cz, db_cl], axis=1).T
# print(combined_data)
# mean_df = combined_data.mean(axis=None)
# print(mean_df)


# combined_data = combined_data.apply(lambda x: x - mean_df)
# # print(combined_data)
# # PCA
# pca = PCA(n_components=6)
# pca_result = pca.fit_transform(df)

# pca_result_df = pd.DataFrame(pca_result)
# print(pca_result_df)


col = ["CZ", "CZ", "CZ", "CL1", "CL1", "CL1"]
fig = px.scatter(pca_result, x=pca_result[:, 0], y=pca_result[:, 1], color=col)
fig.show()

# plt.scatter(pca_result[:, 0], pca_result[:, 1])
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA")
# plt.show()

# import plotly.express as px
# from sklearn.decomposition import PCA

# df = px.data.iris()
# print(df.head)
# X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]


# pca = PCA(n_components=2)
# components = pca.fit_transform(X)
# print(components)

# fig = px.scatter(components, x=0, y=1, color=df["species"])
# fig.show()


## HEAT - NEEDS autoscaling, smpl - mean/stand.dev
# import plotly.express as px

# # fig = go.Figure(go.Heatmap(data=combined_data, colorscale="RdBu", zmid=0))
# fig = px.imshow(
#     combined_data,
#     x=combined_data.columns,
#     y=combined_data.index,
#     color_continuous_scale="RdBu",
# )
# fig.update_layout(width=500, height=500)
# fig.show()
