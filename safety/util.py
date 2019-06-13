import glob
import logging
import os

import imblearn
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# from safety.feature_engineering_pandas import LABEL

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('root')

def read_multiple_csv_pandas(folder_path):
    csv_files = sorted(glob.glob(os.path.join(folder_path, '*.csv')))
    list_of_df = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(list_of_df, ignore_index=True)
    return df

# def random_under_sampling(df: pd.DataFrame, negative_ratio: int = 1):
#     df_positive = df[df[LABEL] == 1]
#     df_negative = df[df[LABEL] == 0]
