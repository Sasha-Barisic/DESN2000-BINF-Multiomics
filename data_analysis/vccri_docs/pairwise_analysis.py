import pandas as pd
from copy import deepcopy
import re
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

def checkout_columns(columns: list):
  cols_idx = {}
  for col in columns:
    if re.match('[a-zA-Z_]+[0-9\.]+$', col):
      pre, i = col.split('.')
      if pre not in cols_idx.keys():
        cols_idx[pre] = [i]
      else:
        cols_idx[pre].append(i)
  for key in cols_idx.keys():
    cols_idx[key].sort()
  blank_cols = [ 'Blank.' + i for i in cols_idx['Blank'] ]
  cols_idx.pop('Blank')
  cx_cols = [ [key + '.' + i for i in cols_idx[key] ] for key in cols_idx.keys()]
  return blank_cols, cx_cols


def clean_first(df: pd.DataFrame) -> pd.DataFrame:
  xdf = deepcopy(df)
  blank_cols, _ = checkout_columns(df.columns)

  for idx, row in xdf.iterrows():
    blank_value = np.array(row[blank_cols])
    first_check = [ blank_value[0] >= bv for bv in 2 * blank_value[1:] ]
    if True in first_check:
      xdf.drop([idx], inplace=True)
  return xdf

def clean_folder_change(df: pd.DataFrame) -> pd.DataFrame:
  xdf = deepcopy(df)
  blank_cols, cx_cols = checkout_columns(df.columns)
  for idx, row in xdf.iterrows():
    for x in cx_cols:
      c_avg = row[x].mean()
      b_avg = row[blank_cols].mean()
      if c_avg == 0 or b_avg / c_avg > 2.0:
        xdf.drop([idx], inplace=True)
        break
  return xdf

def clean_pQ_value(df: pd.DataFrame, cleanQv = False) ->  pd.DataFrame:
  xdf = deepcopy(df)
  blank_cols, cx_cols = checkout_columns(df.columns)
  for idx, row in xdf.iterrows():
    pv = []
    wc = False
    for x in cx_cols:
      _, p = stats.ttest_ind(np.array(row[x].to_list()) ,
                              np.array(row[blank_cols].to_list()))
      if p >= 0.05:
        xdf.drop([idx], inplace=True)
        wc = True
        break
      else:
        pv.append(p)   
    if wc and not cleanQv:
      continue
    if not wc and cleanQv:
      spv = deepcopy(pv)
      spv.sort()
      wc = False
      for i in range(len(pv)):
        for j in range(len(pv)):
          if spv[j] == spv[i]:
            Qv = pv[i] * len(blank_cols) / (j + 1)
            if Qv >= 0.05:
              xdf.drop([idx], inplace=True)
              wc =True
            break
        if wc:
          break

  return xdf


   

def test():
  df = pd.read_csv('annotated_MS_peaks-normalized.csv')
  df = clean_first(df)
  df = clean_folder_change(df)
  df = clean_pQ_value(df, True)
  return df


def pairwise_comparison():
    # clean_df = test()
    df = pd.read_csv('two_samples.csv', sep=',').set_index('unique_id').T
    df.drop(columns=['label'], inplace=True)

    # # Convert pd.df to r.df
    with localconverter(robjects.default_converter + pandas2ri.converter):
        rdf = robjects.conversion.py2rpy(df)

    robjects.globalenv['rdf'] = rdf
    robjects.r("library(randomForest)")
    robjects.r("library(dplyr)")

    # Define the RF.Anal function in Python

    robjects.r(
        f"""
        l_rand <- runif(1)
        rn.sd <- l_rand
        set.seed(rn.sd)

        rf_out <- randomForest::randomForest(rdf, ntree = 500, mtry = 7, importance = TRUE, proximity = TRUE)

        impmat <- rf_out$importance
        impmat <- data.frame(MeanDecreaseAccuracy = impmat[, "MeanDecreaseAccuracy"])
        impmat <- impmat[order(-impmat$MeanDecreaseAccuracy), , drop = FALSE]

        colnames(impmat) <- c("MeanDecreaseAccuracy")
        rownames(impmat) <- NULL

        print(impmat)

    """
    )

    # Convert the result to a pandas DataFrame

    # Convert the result to a numpy array
    rf_table = robjects.globalenv["impmat"]
    with localconverter(robjects.default_converter + pandas2ri.converter) as cv:
        np_from_r_df = robjects.conversion.rpy2py(rf_table)

    # Create a pandas DataFrame with appropriate column names
    column_names = ["MeanDecreaseAccuracy"]
    variable_importance = pd.DataFrame(np_from_r_df, columns=column_names)

    # Sort the DataFrame by MeanDecreaseAccuracy in descending order
    variable_importance = variable_importance.sort_values(by=['MeanDecreaseAccuracy'], ascending=False)

    # Plot the variable importance as a scatter plot
    fig = go.Figure(data=go.Scatter(x=variable_importance['MeanDecreaseAccuracy'], y=variable_importance.index, mode='markers'))
    fig.update_layout(title='Random Forest Variable Importance', xaxis_title='MeanDecreaseAccuracy', yaxis_title='Variable')
    fig.show()

    

#--------------------------------------------------------------------------May not be required--------------------------------------------
    # # Perform PCA
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(data)

    # # Perform K-means clustering
    # kmeans = KMeans(n_clusters=2, random_state=0)
    # kmeans.fit(data)

    # # PCA Plot
    # fig_pca = px.scatter(pca_result, x=0, y=1, color=kmeans.labels_, hover_data=[data.index])
    # fig_pca.update_layout(title='PCA')

    # # K-means Plot

    # # Perform PCA
    # pca = PCA(n_components=2)
    # pca_result = pca.fit_transform(data)

    # # Perform K-means clustering
    # kmeans = KMeans(n_clusters=3)
    # kmeans.fit(data)

    # # Create a dataframe with the PCA results and K-means labels
    # df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    # df['Cluster'] = kmeans.labels_

    # # Format the percentage values for axes labels
    # percentage_labels = lambda x: f'{x:.2f}%'

    # # Plot K-means results using plotly
    # fig = px.scatter(df, x='PC1', y='PC2', color='Cluster', 
    #                 labels={'PC1': 'PC1 ({:.2f}%)'.format(pca.explained_variance_ratio_[0] * 100),
    #                         'PC2': 'PC2 ({:.2f}%)'.format(pca.explained_variance_ratio_[1] * 100)},
    #                 title='K-means Clustering')
    # fig.show()

    # fig_kmeans = px.scatter(pca_result, x=0, y=1, color=kmeans.labels_, hover_data=[data.index])
    # fig_kmeans.update_layout(title='K-means Clustering')

    # # # Create a random forest classifier
    
    
    # db1_cols = list(db1.columns)
    # db2_cols = list(db2.columns)
    
    # db2_modified = db2.rename(columns = {db2_cols[0]:db1_cols[0], db2_cols[1]:db1_cols[1], db2_cols[2]:db1_cols[2]})

    # features = pd.concat([db1, db2_modified], axis=0)
    # labels = ['CZ'] * len(db1) + ['CL1'] * len(db2)

    
    # forest = RandomForestClassifier()

    # # # Fit the classifier to the data
    # forest.fit(data, kmeans.labels_)

    # # Random Forest Feature Importance
    # feature_importances = forest.feature_importances_
    # fig_rf = go.Figure(data=go.Bar(x=data.columns, y=feature_importances))
    # fig_rf.update_layout(title='Random Forest Feature Importance')

    # rf_classifier = RandomForestClassifier()
    # rf_classifier.fit(features, labels)

    # # Extract the main decision tree
    # main_tree = rf_classifier.estimators_[0]

    # # Visualize the main decision tree
    # plt.figure(figsize=(12, 8))
    # tree.plot_tree(main_tree, filled=True, rounded=True)
    # plt.title('Main Decision Tree')
    # plt.show()

    # # Volcano Plot
    # p_values = []
    # for i in range(len(db1)):
    #     cz_values = db1.iloc[i, :].values
    #     cl1_values = db2.iloc[i, :].values
    #     _, p_value = ttest_ind(cz_values, cl1_values)
    #     p_values.append(p_value)

    # fold_change = np.log2(db2.mean(axis=1) / db1.mean(axis=1))
    # significance_threshold = 0.05

    # volcano_df = pd.DataFrame({'Fold Change': fold_change, 'p-value': p_values})
    # volcano_df['Significant'] = (volcano_df['p-value'] < significance_threshold) & (volcano_df['Fold Change'] > 2)

    # # plt.scatter(volcano_df['Fold Change'], -np.log10(volcano_df['p-value']), c=volcano_df['Significant'])
    # fig_volcano = px.scatter(volcano_df, x=volcano_df['Fold Change'], y=-np.log10(volcano_df['p-value']), labels= {"Fold Change": "log2(FC)",
    #                  "y": "-log10(p)" }, color=volcano_df['Significant'])
    # fig_volcano.update_layout(title='Volcano Plot')

    # # Heatmap
    # fig_heatmap = go.Figure(data=go.Heatmap(z=data.values, x=data.columns, y=data.index))
    # fig_heatmap.update_layout(title='Heatmap')

    # # Ortho-PLSDA
    # plsda = PLSRegression(n_components=2)
    # plsda.fit(db1, db2)

    # plsda_scores = plsda.transform(db1)

    # plt.scatter(plsda_scores[:, 0], plsda_scores[:, 1])
    # plt.xlabel('PLS-DA Component 1')
    # plt.ylabel('PLS-DA Component 2')
    # plt.title('Ortho-PLSDA')
    # plt.show()

    # Display the plots
    # fig_volcano.show()
    # fig_kmeans.show()
    # # fig_rf.show()
    # fig_pca.show()
    # fig_heatmap.show()


    # # RandomForest

    # Extract features (X) and labels (y)
    # X = clean_df.drop(['CZ.1', 'CZ.2', 'CZ.3', 'CL1.1', 'CL1.2', 'CL1.3'], axis=1)
    # y = clean_df['CL1.1'].apply(lambda x: 'CZ' if pd.isnull(x) else 'CL1')

    # # Encode labels
    # le = LabelEncoder()
    # labels = le.fit_transform(labels)

    # # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # # Random Forest Classifier
    # forest = RandomForestClassifier(n_estimators=100)
    # forest.fit(X_train, y_train)
    
    # for i, tree_in_forest in enumerate(rf_classifier.estimators_):
    #     plt.figure(figsize=(12, 8))
    #     tree.plot_tree(tree_in_forest, filled=True, rounded=True)
    #     plt.title('Decision Tree {}'.format(i+1))
    #     plt.show()

    # # Feature Importances
    # importances = forest.feature_importances_
    # indices = np.argsort(importances)[::-1]
    # feature_names = features.columns

    # # Plot Feature Importances
    # plt.figure()
    # plt.title("Random Forest Feature Importances")
    # plt.bar(range(features.shape[1]), importances[indices], align='center')
    # plt.xticks(range(features.shape[1]), feature_names[indices], rotation=90)
    # plt.tight_layout()
    # plt.show()

    # # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    # # rf_classifier.fit(features, labels)
    # # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

    # # tree.plot_tree(rf_classifier.estimators_[0],
    # #                 feature_names = features, 
    # #                 class_names=labels,
    # #                 filled = True)
    # #print(len(features))

    # plt.scatter(pca_result[:, 0], pca_result[:, 1])
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('PCA')
    # plt.show()

    # # Heatmap
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(combined_data, cmap='coolwarm')
    # plt.title('Heatmap')
    # plt.show()

    # # K-means
    # kmeans = KMeans(n_clusters=2)
    # kmeans_features = pd.concat([db1, db2], axis=1)
    # kmeans.fit(kmeans_features)

    # plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_)
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title('K-means Clustering')
    # plt.show()

if __name__ == '__main__':
    pairwise_comparison()
    