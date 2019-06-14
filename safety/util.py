import glob
import logging
import os

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

LABEL = 'label'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('root')


def read_multiple_csv_pandas(folder_path):
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
    list_of_df = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(list_of_df, ignore_index=True)
    return df


def random_under_sampling(df: pd.DataFrame, negative_ratio: int = 1):
    positive_count = len(df[df[LABEL] == 1])
    rus = RandomUnderSampler(random_state=999, return_indices=True,
                             ratio={1: positive_count, 0: positive_count * negative_ratio})

    column_list = df.columns
    df_label = df[LABEL]
    df_features = df[column_list[:-1]]

    _, _, idx = rus.fit_sample(df_features, df_label)
    df = df.iloc[idx]
    return df
