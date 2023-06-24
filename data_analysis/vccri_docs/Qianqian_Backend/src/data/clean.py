import pandas as pd
from copy import deepcopy
import re
from scipy import stats
import numpy as np

def data_clean(df: pd.DataFrame) -> pd.DataFrame:
    xdf = deepcopy(df)
    _cols = []
    for col in xdf.columns:
      if re.match('[a-zA-Z_]+[0-9\.]+$', col):
        if not re.match('^Blank.*', col):
          _cols.append(col)
    cols = []
    for col in _cols:
      pre, i = col.split('.')
      pre += '.'
      if i == '1':
        if pre + '2' in _cols and pre + '3' in _cols:
          if [pre + '1', pre + '2' , pre + '3'] not in cols:
            cols.append([pre + '1', pre + '2' , pre + '3'])
      elif i == '2':
        if pre + '1' in _cols and pre + '3' in _cols:
          if [pre + '1', pre + '2' , pre + '3'] not in cols:
            cols.append([pre + '1', pre + '2' , pre + '3'])
      elif i == '3':
        if pre + '2' in _cols and pre + '1' in _cols:
          if [pre + '1', pre + '2' , pre + '3'] not in cols:
            cols.append([pre + '1', pre + '2' , pre + '3'])
    for idx, row in xdf.iterrows():
      pv = []
      if row['Blank.1'] >= 2.0 * row['Blank.2'] and row['Blank.1'] >= 2.0 * row['Blank.3']:
        xdf.drop([idx], inplace=True)
        continue

      wc = False
      for x in cols:
        c_avg = row[x].mean()
        b_avg = row[['Blank.1', 'Blank.2', 'Blank.3']].mean()
        if c_avg == 0 or b_avg / c_avg > 2.0:
          xdf.drop([idx], inplace=True)
          wc = True
          break


        _, p = stats.ttest_ind(np.array([row[x[0]], row[x[1]], row[x[2]]]) ,
                              np.array([row['Blank.1'] , row['Blank.2'], row['Blank.3']]))
        if p >= 0.05:
          xdf.drop([idx], inplace=True)
          wc = True
          break
        else:
          pv.append(p)        

      if wc: 
        continue
      spv = deepcopy(pv)
      spv.sort()
      wc = False
      for i in range(len(pv)):
        for j in range(len(pv)):
          if spv[j] == spv[i]:
            Qv = pv[i] * 3 / (j + 1)
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
  ndf = data_clean(df)
  print(df.shape, ndf.shape)

if __name__ == '__main__':
    test()
    