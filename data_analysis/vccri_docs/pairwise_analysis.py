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
    clean_df = test()
    db1 = clean_df.filter(regex='CZ')
    db2 = clean_df.filter(regex='CL1')

    # Step 2: Perform Pairwise Comparisons
    p_values = []
    for i in range(len(db1)):
        cz_values = db1.iloc[i, :].values
        cl1_values = db2.iloc[i, :].values
        _, p_value = ttest_ind(cz_values, cl1_values)
        p_values.append(p_value)

    # Adjust p-values using the Benjamini-Hochberg method
    #adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

    # Step 3: Visualize the Data

    # Volcano Plot
    fold_change = np.log2(db1.mean(axis=1) / db2.mean(axis=1))
    significance_threshold = 0.05

    volcano_df = pd.DataFrame({'Fold Change': fold_change, 'p-value': p_values})
    volcano_df['Significant'] = volcano_df['p-value'] < significance_threshold

    plt.scatter(volcano_df['Fold Change'], -np.log10(volcano_df['p-value']), c=volcano_df['Significant'])
    plt.xlabel('Fold Change (log2)')
    plt.ylabel('-log10(p-value)')
    plt.title('Volcano Plot')
    plt.show()

    # RandomForest
    db1_cols = list(db1.columns)
    db2_cols = list(db2.columns)
    
    db2_modified = db2.rename(columns = {db2_cols[0]:db1_cols[0], db2_cols[1]:db1_cols[1], db2_cols[2]:db1_cols[2]})

    # features = pd.concat([db1, db2_modified], axis=0)
    # labels = ['CZ'] * len(db1) + ['CL1'] * len(db2)
    # rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    # rf_classifier.fit(features, labels)
    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)

    # tree.plot_tree(rf_classifier.estimators_[0],
    #                 feature_names = features, 
    #                 class_names=labels,
    #                 filled = True)
    #print(len(features))
    #print(len(labels))

    combined_data = pd.concat([db1, db2], axis=1)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_data)

    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA')
    plt.show()

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(combined_data, cmap='coolwarm')
    plt.title('Heatmap')
    plt.show()

    # K-means
    kmeans = KMeans(n_clusters=2)
    kmeans_features = pd.concat([db1, db2], axis=1)
    kmeans.fit(kmeans_features)

    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans.labels_)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-means Clustering')
    plt.show()

    # Ortho-PLSDA
    plsda = PLSRegression(n_components=2)
    plsda.fit(db1, db2)

    plsda_scores = plsda.transform(db1)

    plt.scatter(plsda_scores[:, 0], plsda_scores[:, 1])
    plt.xlabel('PLS-DA Component 1')
    plt.ylabel('PLS-DA Component 2')
    plt.title('Ortho-PLSDA')
    plt.show()

if __name__ == '__main__':
    pairwise_comparison()
    