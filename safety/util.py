import glob
import logging
import os

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

LABEL = 'label'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('root')

LOG1P_COLUMN_LIST = ['total_acceleration_kurt', 'total_acceleration_abs_mean', 'total_acceleration_hilbert_mean',
                     'total_acceleration_hann_mean', 'total_acceleration_std_last_10_sec',
                     'total_acceleration_mad_last_10_sec', 'total_acceleration_std_last_15_sec',
                     'total_acceleration_mad_last_15_sec', 'total_acceleration_std_last_30_sec',
                     'total_acceleration_mad_last_30_sec', 'total_gyro_max', 'total_gyro_mean', 'total_gyro_sum',
                     'total_gyro_mad', 'total_gyro_skew', 'total_gyro_std', 'total_gyro_kurt', 'total_gyro_abs_mean',
                     'total_gyro_hilbert_mean', 'total_gyro_hann_mean', 'total_gyro_max_last_10_sec',
                     'total_gyro_mean_last_10_sec', 'total_gyro_std_last_10_sec', 'total_gyro_mad_last_10_sec',
                     'total_gyro_max_last_15_sec', 'total_gyro_mean_last_15_sec', 'total_gyro_std_last_15_sec',
                     'total_gyro_mad_last_15_sec', 'total_gyro_max_last_30_sec', 'total_gyro_mean_last_30_sec',
                     'total_gyro_std_last_30_sec', 'total_gyro_mad_last_30_sec', 'total_gyro_max_last_45_sec',
                     'total_gyro_mean_last_45_sec', 'total_gyro_std_last_45_sec', 'total_gyro_mad_last_45_sec',
                     'total_gyro_max_last_60_sec', 'total_gyro_mean_last_60_sec', 'total_gyro_std_last_60_sec',
                     'total_gyro_mad_last_60_sec', 'total_gyro_max_last_90_sec', 'total_gyro_mean_last_90_sec',
                     'total_gyro_std_last_90_sec', 'total_gyro_mad_last_90_sec', 'total_gyro_max_last_120_sec',
                     'total_gyro_mean_last_120_sec', 'total_gyro_std_last_120_sec', 'total_gyro_mad_last_120_sec',
                     'Speed_min', 'Speed_mean', 'Speed_sum', 'Speed_median', 'Speed_hilbert_mean', 'Speed_hann_mean',
                     'Speed_min_last_10_sec', 'Speed_max_last_10_sec', 'Speed_mean_last_10_sec',
                     'Speed_std_last_10_sec', 'Speed_mad_last_10_sec', 'Speed_min_last_15_sec', 'Speed_max_last_15_sec',
                     'Speed_mean_last_15_sec', 'Speed_std_last_15_sec', 'Speed_mad_last_15_sec',
                     'Speed_min_last_30_sec', 'Speed_max_last_30_sec', 'Speed_mean_last_30_sec',
                     'Speed_std_last_30_sec', 'Speed_mad_last_30_sec', 'Speed_min_last_45_sec', 'Speed_min_last_60_sec',
                     'Speed_min_last_90_sec', 'Speed_min_last_120_sec', 'Bearing_min', 'Bearing_max', 'Bearing_kurt',
                     'Bearing_std_last_10_sec', 'Bearing_mad_last_10_sec', 'Bearing_min_last_10_sec',
                     'Bearing_max_last_10_sec', 'Bearing_std_last_15_sec', 'Bearing_mad_last_15_sec',
                     'Bearing_min_last_15_sec', 'Bearing_max_last_15_sec', 'Bearing_std_last_30_sec',
                     'Bearing_mad_last_30_sec', 'Bearing_min_last_30_sec', 'Bearing_max_last_30_sec',
                     'Bearing_std_last_45_sec', 'Bearing_mad_last_45_sec', 'Bearing_min_last_45_sec',
                     'Bearing_max_last_45_sec', 'Bearing_std_last_60_sec', 'Bearing_mad_last_60_sec',
                     'Bearing_min_last_60_sec', 'Bearing_max_last_60_sec', 'Bearing_std_last_90_sec',
                     'Bearing_mad_last_90_sec', 'Bearing_min_last_90_sec', 'Bearing_max_last_90_sec',
                     'Bearing_std_last_120_sec', 'Bearing_mad_last_120_sec', 'Bearing_min_last_120_sec',
                     'Bearing_max_last_120_sec']

TOP_100_FEATURE_IMPORTANCE = ['second_abs_mean', 'second_median', 'q099_total_gyro', 'second_mean',
                              'second_mean_last_10_sec', 'second_min_last_10_sec', 'Speed_hann_mean', 'second_std',
                              'second_max_last_10_sec', 'total_acceleration_std', 'second_max', 'acceleration_y_mad',
                              'second_mean_last_30_sec', 'second_mad', 'second_min_last_15_sec',
                              'second_mad_last_15_sec', 'acceleration_z_mad', 'MA90_Speed_mean',
                              'second_min_last_120_sec', 'MA60_Speed_mean', 'Accuracy_max_last_120_sec', 'Bearing_std',
                              'Speed_skew', 'Speed_mean', 'second_mad_last_60_sec', 'Speed_iqr', 'Bearing_mad',
                              'q095_total_gyro', 'total_gyro_mad_last_60_sec', 'gyro_z_abs_mean',
                              'second_std_last_60_sec', 'total_gyro_max_last_90_sec', 'Speed_max_last_120_sec',
                              'Speed_median', 'second_mad_last_10_sec', 'acceleration_x_mad', 'acceleration_y_mean',
                              'total_gyro_mad_last_120_sec', 'acceleration_x_std', 'acceleration_y_abs_mean',
                              'total_gyro_max_last_30_sec', 'second_mad_last_30_sec', 'gyro_y_max_last_30_sec',
                              'total_acceleration_mean_change', 'acceleration_z_max_last_60_sec', 'MA10_total_gyro_std',
                              'gyro_y_max', 'Speed_mean_last_120_sec', 'gyro_x_std', 'second_min_last_90_sec',
                              'Accuracy_max_last_10_sec', 'gyro_x_mad', 'gyro_y_mad_last_60_sec', 'Speed_sum',
                              'Speed_abs_mean', 'Speed_max', 'second_skew', 'second_min_last_45_sec',
                              'acceleration_y_median', 'gyro_x_std_last_120_sec', 'second_std_last_120_sec',
                              'acceleration_z_std', 'Bearing_min', 'total_gyro_std_last_30_sec',
                              'second_mad_last_120_sec', 'acceleration_x_std_last_15_sec',
                              'acceleration_z_std_last_45_sec', 'total_acceleration_mean',
                              'acceleration_y_std_last_10_sec', 'acceleration_y_min_last_45_sec', 'gyro_y_abs_mean',
                              'acceleration_y_max_last_15_sec', 'gyro_x_mad_last_90_sec',
                              'acceleration_z_max_last_45_sec', 'acceleration_x_mad_last_15_sec',
                              'total_acceleration_mad_last_90_sec', 'gyro_y_max_last_60_sec', 'total_gyro_mean_change',
                              'Speed_hilbert_mean', 'Bearing_mad_last_10_sec', 'Accuracy_median', 'Speed_trend',
                              'acceleration_x_std_last_30_sec', 'second_std_last_15_sec', 'gyro_y_mad_last_30_sec',
                              'gyro_z_max_last_120_sec', 'MA30_Speed_std', 'second_std_last_90_sec',
                              'Speed_mad_last_60_sec', 'Speed_mean_last_15_sec', 'second_sum',
                              'total_gyro_hilbert_mean', 'acceleration_y_mean_last_10_sec', 'MA10_Speed_mean',
                              'total_acceleration_mad_last_15_sec', 'Bearing_max_last_15_sec',
                              'Accuracy_min_last_15_sec', 'MA60_Speed_std', 'acceleration_z_max_last_120_sec',
                              'Speed_std_last_45_sec']


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
