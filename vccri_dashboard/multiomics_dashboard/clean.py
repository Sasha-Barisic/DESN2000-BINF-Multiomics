import pandas as pd
from copy import deepcopy
import re
import numpy as np
from scipy import stats


def checkout_columns(columns: list):
    cols_idx = {}
    cols_to_drop = []
    id_cols = []
    for col in columns:
        if re.search(r"\.([0-9]+)$", col):
            pre, i = col.rsplit(".", 1)
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
            else:
                id_cols.append(col)

    # Sort the entries in the dictionary.
    for key in cols_idx.keys():
        cols_idx[key].sort()

    # Separate the blanks from the samples.
    blank_cols = [i for i in cols_idx["Blank"]]

    cols_idx.pop("Blank")
    cx_cols = [[i for i in cols_idx[key]] for key in cols_idx.keys()]

    return cols_to_drop, blank_cols, cx_cols, id_cols


def clean_first(df: pd.DataFrame) -> pd.DataFrame:
    xdf = deepcopy(df)
    del_cols, blank_cols, _, id_cols = checkout_columns(df.columns)

    for idx, row in xdf.iterrows():
        blank_value = np.array(row[blank_cols])
        first_check = [blank_value[0] >= bv for bv in 2 * blank_value[1:]]

        if True in first_check:
            xdf.drop([idx], inplace=True)

    cleaned_df = xdf.drop(columns=del_cols)

    # Create unique-ID
    unique_ids = []

    for _, row in cleaned_df.iterrows():
        if pd.isna(row[id_cols[0]]):
            unique_ids.append(row[id_cols[1]])
        elif pd.isna(row[id_cols[1]]):
            unique_ids.append(row[id_cols[0]])
        else:
            id = f"{row[id_cols[0]]}-{row[id_cols[1]]}"
            unique_ids.append(id)

    cleaned_df["unique_id"] = unique_ids
    cleaned_df = cleaned_df.drop(columns=id_cols)

    first_column = cleaned_df.pop("unique_id")
    cleaned_df.insert(0, "unique_id", first_column)

    return cleaned_df


def clean_folder_change(df: pd.DataFrame) -> pd.DataFrame:
    xdf = deepcopy(df)
    del_cols, blank_cols, cx_cols, _ = checkout_columns(df.columns)

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
    del_cols, blank_cols, cx_cols, _ = checkout_columns(df.columns)

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

    # Adding the label row to the df.
    labels = []
    samples = list(xdf.columns)
    for smpl in samples:
        if smpl == "unique_id":
            labels.append("label")
        else:
            labels.append(smpl.split(".")[0])

    new_xdf = pd.DataFrame(labels).T
    new_xdf.columns = samples
    new_xdf = pd.concat([new_xdf, xdf])
    new_xdf = new_xdf.reset_index(drop=True)

    # # extract from the new_xdf two samples
    # db1 = new_xdf.filter(regex="CZ|CL1")
    # db1.to_csv("two_samples.csv", sep=",")
    return new_xdf
