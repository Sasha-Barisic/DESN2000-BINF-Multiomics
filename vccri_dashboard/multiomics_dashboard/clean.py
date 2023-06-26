import pandas as pd
from copy import deepcopy
import re
import numpy as np
from scipy import stats


def checkout_columns(columns: list):
    cols_idx = {}
    cols_to_drop = []

    for col in columns:
        # Columns that match this format XXN(_X).N
        if re.match("[a-zA-Z0-9_]+\.[0-9]+$", col):
            pre, i = col.split(".")

            if pre not in cols_idx.keys():
                cols_idx[pre] = [col]
            else:
                cols_idx[pre].append(col)

        # Collect all the other besides "structure" and "selected_feature".
        else:
            if not re.search("structure", col, re.IGNORECASE) and not re.search(
                "selected_feature", col, re.IGNORECASE
            ):
                cols_to_drop.append(col)

    # Sort the entries in the dictionary.
    for key in cols_idx.keys():
        cols_idx[key].sort()

    # Separate the blanks from the samples.
    blank_cols = [i for i in cols_idx["Blank"]]

    cols_idx.pop("Blank")
    cx_cols = [[i for i in cols_idx[key]] for key in cols_idx.keys()]

    return cols_to_drop, blank_cols, cx_cols


def clean_first(df: pd.DataFrame) -> pd.DataFrame:
    xdf = deepcopy(df)
    del_cols, blank_cols, _ = checkout_columns(df.columns)

    for idx, row in xdf.iterrows():
        blank_value = np.array(row[blank_cols])
        first_check = [blank_value[0] >= bv for bv in 2 * blank_value[1:]]

        if True in first_check:
            xdf.drop([idx], inplace=True)

    cleaned_df = xdf.drop(columns=del_cols)

    return cleaned_df


def clean_folder_change(df: pd.DataFrame) -> pd.DataFrame:
    xdf = deepcopy(df)
    del_cols, blank_cols, cx_cols = checkout_columns(df.columns)

    for idx, row in xdf.iterrows():
        for x in cx_cols:
            c_avg = row[x].mean()
            b_avg = row[blank_cols].mean()

            if c_avg == 0 or b_avg / c_avg > 2.0:
                xdf.drop([idx], inplace=True)

                break
    return xdf


def clean_pQ_value(df: pd.DataFrame, cleanQv=False) -> pd.DataFrame:
    xdf = deepcopy(df)
    del_cols, blank_cols, cx_cols = checkout_columns(df.columns)

    for idx, row in xdf.iterrows():
        pv = []
        wc = False

        for x in cx_cols:
            _, p = stats.ttest_ind(
                np.array(row[x].to_list()), np.array(row[blank_cols].to_list())
            )

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
                            wc = True
                        break
                if wc:
                    break

    return xdf
