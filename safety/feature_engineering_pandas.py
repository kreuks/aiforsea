import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

BOOKING_ID = 'bookingID'
ACCURACY = 'Accuracy'
BEARING = 'Bearing'
ACC_X = 'acceleration_x'
ACC_Y = 'acceleration_y'
ACC_Z = 'acceleration_z'
GYRO_X = 'gyro_x'
GYRO_Y = 'gyro_y'
GYRO_Z = 'gyro_z'
SECOND = 'second'
SPEED = 'Speed'

TOTAL_ACC = 'total_acceleration'
TOTAL_GYRO = 'total_gyro'
SPEED_TREND = SPEED + '_trend'
TOTAL_ACC_TREND = TOTAL_ACC + '_trend'
TOTAL_GYRO_TREND = TOTAL_GYRO + '_trend'

TOTAL_ACC_MEAN_CHANGE = TOTAL_ACC + '_mean_change'
TOTAL_GYRO_MEAN_CHANGE = TOTAL_GYRO + '_mean_change'
SPEED_MEAN_CHANGE = SPEED + '_mean_change'

TOTAL_ACC_MEAN_CHANGE_RATE = TOTAL_ACC + '_mean_change_rate'
TOTAL_GYRO_MEAN_CHANGE_RATE = TOTAL_GYRO + '_mean_change_rate'
SPEED_MEAN_CHANGE_RATE = SPEED + '_mean_change_rate'

TOTAL_ACC_HILBERT_MEAN = TOTAL_ACC + '_hilbert_mean'
TOTAL_ACC_HANN_MEAN = TOTAL_ACC + '_hann_mean'
TOTAL_GYRO_HILBERT_MEAN = TOTAL_GYRO + '_hilbert_mean'
TOTAL_GYRO_HANN_MEAN = TOTAL_GYRO + '_hann_mean'
SPEED_HILBERT_MEAN = SPEED + '_hilbert_mean'
SPEED_HANN_MEAN = SPEED + '_hann_mean'

TOTAL_ACC_IQR = TOTAL_ACC + '_iqr'
TOTAL_GYRO_IQR = TOTAL_GYRO + '_iqr'
SPEED_IQR = SPEED + '_iqr'

LABEL = 'label'


class FeatureEngineering:

    def __get_agg_last_n_sec(self, df, n):
        logging.info('start agg_for_last_{}_sec'.format(n))

        df = df.sort_values([BOOKING_ID, SECOND], ascending=[True, False])
        df = df.groupby(BOOKING_ID).head(n).reset_index(drop=True)

        df_mean_change_rate = self.__get_mean_change_and_mean_change_rate_df(df, '_last_{}_sec'.format(n))
        df = df.groupby(BOOKING_ID).agg(['min', 'max', 'mean', 'std', 'mad', 'skew'])
        df.columns = ['_'.join(tup) + '_last_{}_sec'.format(n) for tup in df.columns.values]
        df = df.reset_index()

        df = pd.merge(df, df_mean_change_rate, on=BOOKING_ID, how='left')

        logging.info('finish agg_for_last_{}_sec'.format(n))
        return df

    def __get_agg_last_10_sec(self, df):
        return self.__get_agg_last_n_sec(df, 10)

    def __get_agg_last_15_sec(self, df):
        return self.__get_agg_last_n_sec(df, 15)

    def __get_agg_last_30_sec(self, df):
        return self.__get_agg_last_n_sec(df, 30)

    def __get_agg_last_45_sec(self, df):
        return self.__get_agg_last_n_sec(df, 45)

    def __get_agg_last_60_sec(self, df):
        return self.__get_agg_last_n_sec(df, 60)

    def __get_agg_last_90_sec(self, df):
        return self.__get_agg_last_n_sec(df, 90)

    def __get_agg_last_120_sec(self, df):
        return self.__get_agg_last_n_sec(df, 120)

    def __get_primitive_aggregation_df(self, df: pd.DataFrame):
        logging.info('starting group by primitive')

        df = df.groupby(BOOKING_ID).agg(['min', 'max', 'mean', 'sum', 'mad', 'skew', 'median', 'std'])
        df.columns = ['_'.join(tup) for tup in df.columns.values]
        df = df.reset_index()

        logging.info('finish group by primitive')
        return df

    def __get_kurtosis_aggregation_df(self, df: pd.DataFrame):
        logging.info('starting group by kurtosis')

        df = df.groupby(BOOKING_ID).apply(pd.DataFrame.kurt).drop(BOOKING_ID, axis=1)
        df.columns = ['{}_kurt'.format(x) for x in df.columns]
        df = df.reset_index()

        logging.info('finish group by kurtosis')
        return df

    def __get_mean_absolute_value_df(self, df: pd.DataFrame):
        logging.info('starting get mean absolute')

        df = df.apply(abs)
        df = df.groupby(BOOKING_ID).agg('mean')
        df.columns = ['{}_abs_mean'.format(x) for x in df.columns]
        df = df.reset_index()

        logging.info('finish get mean absolute')
        return df

    def __get_total_acceleration_and_gyro_df(self, df: pd.DataFrame):
        logging.info('starting count total acceleration and gyro')

        df[TOTAL_ACC] = np.sqrt((df[ACC_X] ** 2) + (df[ACC_Y] ** 2) + (df[ACC_Z] ** 2))
        df[TOTAL_GYRO] = np.sqrt((df[GYRO_X] ** 2) + (df[GYRO_Y] ** 2) + (df[GYRO_Z] ** 2))

        logging.info('finish count total acceleration and gyro')
        return df

    def __get_trend_features_df(self, df: pd.DataFrame):
        def __calc_linear_regression_coef(df: pd.DataFrame, col):
            lr = LinearRegression()

            seconds = df[SECOND].values
            seconds = seconds.reshape(-1, 1)
            target = df[col].values
            lr.fit(seconds, target)
            return lr.coef_[0]

        logging.info('starting trend feature total acc')
        df_acc = df.groupby(BOOKING_ID).apply(__calc_linear_regression_coef, TOTAL_ACC).reset_index()
        df_acc.columns = [BOOKING_ID, TOTAL_ACC_TREND]
        logging.info('finish trend feature total acc')

        logging.info('starting trend feature total gyro')
        df_gyro = df.groupby(BOOKING_ID).apply(__calc_linear_regression_coef, TOTAL_GYRO).reset_index()
        df_gyro.columns = [BOOKING_ID, TOTAL_GYRO_TREND]
        logging.info('finish trend feature total gyro')

        logging.info('starting trend feature speed')
        df_speed = df.groupby(BOOKING_ID).apply(__calc_linear_regression_coef, SPEED).reset_index()
        df_speed.columns = [BOOKING_ID, SPEED_TREND]
        logging.info('finish trend feature speed')

        df = pd.merge(df_acc, df_gyro, on=BOOKING_ID, how='left')
        df = pd.merge(df, df_speed, on=BOOKING_ID, how='left')

        return df

    def __get_mean_change_and_mean_change_rate_df(self, df: pd.DataFrame, additional_col_name=''):
        def __calc_change_rate(df: pd.DataFrame, col):
            arr = df[col].values
            change = np.diff(arr) / arr[:-1]
            change = change[np.nonzero(change)[0]]
            change = np.mean(change)
            return change

        logging.info('start mean change rate acc')
        df_acc_change_rate = df.groupby(BOOKING_ID).apply(__calc_change_rate, TOTAL_ACC).reset_index()
        df_acc_change_rate.columns = [BOOKING_ID, TOTAL_ACC_MEAN_CHANGE_RATE + additional_col_name]
        logging.info('finish mean change rate acc')

        logging.info('start mean change rate gyro')
        df_gyro_change_rate = df.groupby(BOOKING_ID).apply(__calc_change_rate, TOTAL_GYRO).reset_index()
        df_gyro_change_rate.columns = [BOOKING_ID, TOTAL_GYRO_MEAN_CHANGE_RATE + additional_col_name]
        logging.info('finish mean change rate gyro')

        logging.info('start mean change rate speed')
        df_speed_change_rate = df.groupby(BOOKING_ID).apply(__calc_change_rate, SPEED).reset_index()
        df_speed_change_rate.columns = [BOOKING_ID, SPEED_MEAN_CHANGE_RATE + additional_col_name]
        logging.info('finish mean change rate speed')

        logging.info('start mean change acc')
        df_acc_mean_change = df[[BOOKING_ID, TOTAL_ACC]].groupby(BOOKING_ID).apply(np.diff).apply(np.mean).reset_index()
        df_acc_mean_change.columns = [BOOKING_ID, TOTAL_ACC_MEAN_CHANGE + additional_col_name]
        logging.info('finish mean change acc')

        logging.info('start mean change gyro')
        df_gyro_mean_change = df[[BOOKING_ID, TOTAL_GYRO]].groupby(BOOKING_ID).apply(np.diff).apply(
            np.mean).reset_index()
        df_gyro_mean_change.columns = [BOOKING_ID, TOTAL_GYRO_MEAN_CHANGE + additional_col_name]
        logging.info('finish mean change gyro')

        logging.info('start mean change speed')
        df_speed_mean_change = df[[BOOKING_ID, SPEED]].groupby(BOOKING_ID).apply(np.diff).apply(np.mean).reset_index()
        df_speed_mean_change.columns = [BOOKING_ID, SPEED_MEAN_CHANGE + additional_col_name]
        logging.info('finish mean change gyro')

        df_list = [df_acc_change_rate,
                   df_gyro_change_rate,
                   df_speed_change_rate,
                   df_acc_mean_change,
                   df_gyro_mean_change,
                   df_speed_mean_change]

        df = df_list[0]

        for df_ in df_list[1:]:
            df = pd.merge(df, df_, on=BOOKING_ID, how='left')

        return df

    def __get_hilbert_mean_and_hann_window_mean_df(self, df: pd.DataFrame):
        logging.info('starting hilbert mean and hann window mean')

        def __calc_hilbert_mean(df: pd.DataFrame, col):
            arr = df[col].values
            return np.abs(hilbert(arr)).mean()

        def __calc_hann_window_mean(df: pd.DataFrame, col):
            arr = df[col].values
            hann_window = convolve(arr, hann(10), mode='same') / sum(hann(10))
            return hann_window.mean()

        df_acc_hilbert_mean = df[[BOOKING_ID, TOTAL_ACC]].groupby(BOOKING_ID).apply(__calc_hilbert_mean,
                                                                                    TOTAL_ACC).reset_index()
        df_acc_hilbert_mean.columns = [BOOKING_ID, TOTAL_ACC_HILBERT_MEAN]
        df_acc_hann_mean = df[[BOOKING_ID, TOTAL_ACC]].groupby(BOOKING_ID).apply(__calc_hann_window_mean,
                                                                                 TOTAL_ACC).reset_index()
        df_acc_hann_mean.columns = [BOOKING_ID, TOTAL_ACC_HANN_MEAN]

        df_gyro_hilbert_mean = df[[BOOKING_ID, TOTAL_GYRO]].groupby(BOOKING_ID).apply(__calc_hilbert_mean,
                                                                                      TOTAL_GYRO).reset_index()
        df_gyro_hilbert_mean.columns = [BOOKING_ID, TOTAL_GYRO_HILBERT_MEAN]
        df_gyro_hann_mean = df[[BOOKING_ID, TOTAL_GYRO]].groupby(BOOKING_ID).apply(__calc_hann_window_mean,
                                                                                   TOTAL_GYRO).reset_index()
        df_gyro_hann_mean.columns = [BOOKING_ID, TOTAL_GYRO_HANN_MEAN]

        df_speed_hilbert_mean = df[[BOOKING_ID, SPEED]].groupby(BOOKING_ID).apply(__calc_hilbert_mean,
                                                                                  SPEED).reset_index()
        df_speed_hilbert_mean.columns = [BOOKING_ID, SPEED_HILBERT_MEAN]
        df_speed_hann_mean = df[[BOOKING_ID, SPEED]].groupby(BOOKING_ID).apply(__calc_hann_window_mean,
                                                                               SPEED).reset_index()
        df_speed_hann_mean.columns = [BOOKING_ID, SPEED_HANN_MEAN]

        df_list = [df_acc_hilbert_mean,
                   df_acc_hann_mean,
                   df_gyro_hilbert_mean,
                   df_gyro_hann_mean,
                   df_speed_hilbert_mean,
                   df_speed_hann_mean]

        df = df_list[0]

        for df_ in df_list[1:]:
            df = pd.merge(df, df_, on=BOOKING_ID, how='left')

        logging.info('finish hilbert mean and hann window mean')

        return df

    def __get_moving_average_mean_df(self, df: pd.DataFrame):
        logging.info('starting ma mean')

        col_list = [BOOKING_ID, TOTAL_ACC, TOTAL_GYRO, SPEED]
        df = df[col_list]
        df_list = []

        rolling_windows = [10, 30, 60, 90]
        for window in rolling_windows:
            temp = df.groupby(BOOKING_ID).rolling(window).mean()
            temp = temp.drop(BOOKING_ID, axis=1).reset_index()[col_list]
            temp = temp.groupby(BOOKING_ID).mean().reset_index()
            temp.columns = [BOOKING_ID] + ['MA{}_{}_mean'.format(window, x) for x in col_list[1:]]
            df_list.append(temp)

        df = df_list[0]
        for df_ in df_list[1:]:
            df = pd.merge(df, df_, on=BOOKING_ID, how='left')

        logging.info('finish ma mean')
        return df

    def __get_moving_average_std_df(self, df: pd.DataFrame):
        logging.info('starting ma std')

        col_list = [BOOKING_ID, TOTAL_ACC, TOTAL_GYRO, SPEED]
        df = df[col_list]
        df_list = []

        rolling_windows = [10, 30, 60, 90]
        for window in rolling_windows:
            temp = df.groupby(BOOKING_ID).rolling(window).std()
            temp = temp.drop(BOOKING_ID, axis=1).reset_index()[col_list]
            temp = temp.groupby(BOOKING_ID).std().reset_index()
            temp.columns = [BOOKING_ID] + ['MA{}_{}_std'.format(window, x) for x in col_list[1:]]
            df_list.append(temp)

        df = df_list[0]
        for df_ in df_list[1:]:
            df = pd.merge(df, df_, on=BOOKING_ID, how='left')

        logging.info('finish ma std')
        return df

    def __get_quantile_df(self, df: pd.DataFrame):
        logging.info('starting quantile df')

        col_list = [BOOKING_ID, TOTAL_ACC, TOTAL_GYRO, SPEED]
        df = df[col_list]
        df_list = []

        quantile_list = [.01, .05, .95, .99]
        for quantile_ in quantile_list:
            temp = df.groupby(BOOKING_ID).quantile(quantile_)
            del temp.columns.name
            temp = temp.reset_index()
            temp.columns = [BOOKING_ID] + ['q{}_{}'.format(quantile_, x).replace('.', '') for x in col_list[1:]]
            df_list.append(temp)

        df = df_list[0]
        for df_ in df_list[1:]:
            df = pd.merge(df, df_, on=BOOKING_ID, how='left')

        logging.info('finish quantile df')
        return df

    def __get_iqr_df(self, df: pd.DataFrame):
        logging.info('starting iqr df')

        col_list = [BOOKING_ID, TOTAL_ACC, TOTAL_GYRO, SPEED]

        df = df[col_list]
        df_25 = df.groupby(BOOKING_ID).quantile(.25).reset_index()
        df_75 = df.groupby(BOOKING_ID).quantile(.75).reset_index()

        df_25.columns = [BOOKING_ID] + ['{}_25'.format(x) for x in col_list[1:]]
        df_75.columns = [BOOKING_ID] + ['{}_75'.format(x) for x in col_list[1:]]

        df = pd.merge(df_25, df_75, on=BOOKING_ID, how='left')
        df[TOTAL_ACC_IQR] = df[TOTAL_ACC + '_75'] - df[TOTAL_ACC + '_25']
        df[TOTAL_GYRO_IQR] = df[TOTAL_GYRO + '_75'] - df[TOTAL_GYRO + '_25']
        df[SPEED_IQR] = df[SPEED + '_75'] - df[SPEED + '_25']

        df = df[[BOOKING_ID, TOTAL_ACC_IQR, TOTAL_GYRO_IQR, SPEED_IQR]]

        logging.info('finish iqr df')
        return df

    def transform(self, df: pd.DataFrame):
        df_label = df.groupby(BOOKING_ID).agg({LABEL: 'last'}).reset_index()

        df: pd.DataFrame = df.drop(LABEL, axis=1)
        df = self.__get_total_acceleration_and_gyro_df(df)

        list_of_df = [self.__get_primitive_aggregation_df(df),
                      self.__get_kurtosis_aggregation_df(df),
                      self.__get_mean_absolute_value_df(df),
                      self.__get_trend_features_df(df),
                      self.__get_mean_change_and_mean_change_rate_df(df),
                      self.__get_hilbert_mean_and_hann_window_mean_df(df),
                      self.__get_agg_last_10_sec(df),
                      self.__get_agg_last_15_sec(df),
                      self.__get_agg_last_30_sec(df),
                      self.__get_agg_last_45_sec(df),
                      self.__get_agg_last_60_sec(df),
                      self.__get_agg_last_90_sec(df),
                      self.__get_agg_last_120_sec(df),
                      self.__get_moving_average_mean_df(df),
                      self.__get_moving_average_std_df(df),
                      self.__get_iqr_df(df),
                      self.__get_quantile_df(df),
                      df_label]

        df = list_of_df[0]
        for df_ in list_of_df[1:]:
            df = pd.merge(df, df_, on=BOOKING_ID, how='left')

        return df