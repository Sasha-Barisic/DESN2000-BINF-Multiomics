import pandas as pd
from copy import deepcopy
import re
from scipy import stats
import numpy as np

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
  df = pd.read_excel('annotated_MS_peaks-normalized.xlsx')
  print(df.shape)
  df = clean_first(df)
  print(df.shape)

  df = clean_folder_change(df)
  print(df.shape)

  # df = clean_pQ_value(df, True)
  # print(df.shape)

  df = clean_pQ_value(df, True)
  print(df.shape)

if __name__ == '__main__':
    test()
    