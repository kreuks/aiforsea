import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

LABEL = 'label'


class FeatureEngineering:

    def __get_agg_last_n_sec(self, df, n):
        print('start agg_for_last_{}_sec'.format(n))

        df = df.sort_values([BOOKING_ID, SECOND], ascending=[True, False])
        df = df.groupby(BOOKING_ID).head(n).reset_index(drop=True)

        df_mean_change_rate = self.__get_mean_change_and_mean_change_rate(df, '_last_{}_sec'.format(n))
        df = df.groupby(BOOKING_ID).agg(['min', 'max', 'mean', 'std', 'mad', 'skew'])
        df.columns = ['_'.join(tup) + '_last_{}_sec'.format(n) for tup in df.columns.values]
        df = df.reset_index()

        df = pd.merge(df, df_mean_change_rate, on=BOOKING_ID, how='left')

        print('finish agg_for_last_{}_sec'.format(n))
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
        print('starting group by primitive')

        df = df.groupby(BOOKING_ID).agg(['min', 'max', 'mean', 'sum', 'mad', 'skew', 'median', 'std'])
        df.columns = ['_'.join(tup) for tup in df.columns.values]
        df = df.reset_index()

        print('finish group by primitive')
        return df

    def __get_kurtosis_aggregation_df(self, df: pd.DataFrame):
        print('starting group by kurtosis')

        df = df.groupby(BOOKING_ID).apply(pd.DataFrame.kurt).drop(BOOKING_ID, axis=1)
        df.columns = ['{}_kurt'.format(x) for x in df.columns]
        df = df.reset_index()

        print('finish group by kurtosis')
        return df

    def __get_mean_absolute_value_df(self, df: pd.DataFrame):
        print('starting get mean absolute')

        df = df.apply(abs)
        df = df.groupby(BOOKING_ID).agg('mean')
        df.columns = ['{}_abs_mean'.format(x) for x in df.columns]
        df = df.reset_index()

        print('finish get mean absolute')
        return df

    def __get_total_acceleration_and_gyro_df(self, df: pd.DataFrame):
        print('starting count total acceleration and gyro')

        df[TOTAL_ACC] = np.sqrt((df[ACC_X] ** 2) + (df[ACC_Y] ** 2) + (df[ACC_Z] ** 2))
        df[TOTAL_GYRO] = np.sqrt((df[GYRO_X] ** 2) + (df[GYRO_Y] ** 2) + (df[GYRO_Z] ** 2))

        print('finish count total acceleration and gyro')
        return df

    def __get_trend_features_df(self, df: pd.DataFrame):
        def __get_linear_regression_coef(df: pd.DataFrame, col):
            lr = LinearRegression()

            seconds = df[SECOND].values
            seconds = seconds.reshape(-1, 1)
            target = df[col].values
            lr.fit(seconds, target)
            return lr.coef_[0]

        print('starting trend feature total acc')
        df_acc = df.groupby(BOOKING_ID).apply(__get_linear_regression_coef, TOTAL_ACC).reset_index()
        df_acc.columns = [BOOKING_ID, TOTAL_ACC_TREND]
        print('finish trend feature total acc')

        print('starting trend feature total gyro')
        df_gyro = df.groupby(BOOKING_ID).apply(__get_linear_regression_coef, TOTAL_GYRO).reset_index()
        df_gyro.columns = [BOOKING_ID, TOTAL_GYRO_TREND]
        print('finish trend feature total gyro')

        print('starting trend feature speed')
        df_speed = df.groupby(BOOKING_ID).apply(__get_linear_regression_coef, SPEED).reset_index()
        df_speed.columns = [BOOKING_ID, SPEED_TREND]
        print('finish trend feature speed')

        df = pd.merge(df_acc, df_gyro, on=BOOKING_ID, how='left')
        df = pd.merge(df, df_speed, on=BOOKING_ID, how='left')

        return df

    def __get_mean_change_and_mean_change_rate(self, df: pd.DataFrame, additional_col_name=''):
        def __calc_change_rate(df: pd.DataFrame, col):
            arr = df[col].values
            change = np.diff(arr) / arr[:-1]
            change = change[np.nonzero(change)[0]]
            change = np.mean(change)
            return change

        print('start mean change rate acc')
        df_acc_change_rate = df.groupby(BOOKING_ID).apply(__calc_change_rate, TOTAL_ACC).reset_index()
        df_acc_change_rate.columns = [BOOKING_ID, TOTAL_ACC_MEAN_CHANGE_RATE + additional_col_name]
        print('finish mean change rate acc')

        print('start mean change rate gyro')
        df_gyro_change_rate = df.groupby(BOOKING_ID).apply(__calc_change_rate, TOTAL_GYRO).reset_index()
        df_gyro_change_rate.columns = [BOOKING_ID, TOTAL_GYRO_MEAN_CHANGE_RATE + additional_col_name]
        print('finish mean change rate gyro')

        print('start mean change rate speed')
        df_speed_change_rate = df.groupby(BOOKING_ID).apply(__calc_change_rate, SPEED).reset_index()
        df_speed_change_rate.columns = [BOOKING_ID, SPEED_MEAN_CHANGE_RATE + additional_col_name]
        print('finish mean change rate speed')

        print('start mean change acc')
        df_acc_mean_change = df[[BOOKING_ID, TOTAL_ACC]].groupby(BOOKING_ID).apply(np.diff).apply(np.mean).reset_index()
        df_acc_mean_change.columns = [BOOKING_ID, TOTAL_ACC_MEAN_CHANGE + additional_col_name]
        print('finish mean change acc')

        print('start mean change gyro')
        df_gyro_mean_change = df[[BOOKING_ID, TOTAL_GYRO]].groupby(BOOKING_ID).apply(np.diff).apply(
            np.mean).reset_index()
        df_gyro_mean_change.columns = [BOOKING_ID, TOTAL_GYRO_MEAN_CHANGE + additional_col_name]
        print('finish mean change gyro')

        print('start mean change speed')
        df_speed_mean_change = df[[BOOKING_ID, SPEED]].groupby(BOOKING_ID).apply(np.diff).apply(np.mean).reset_index()
        df_speed_mean_change.columns = [BOOKING_ID, SPEED_MEAN_CHANGE + additional_col_name]
        print('finish mean change gyro')

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

    def transform(self, df: pd.DataFrame):
        df_label = df.groupby(BOOKING_ID).agg({LABEL: 'last'}).reset_index()

        df: pd.DataFrame = df.drop(LABEL, axis=1)
        df = self.__get_total_acceleration_and_gyro_df(df)

        list_of_df = [self.__get_primitive_aggregation_df(df),
                      self.__get_kurtosis_aggregation_df(df),
                      self.__get_mean_absolute_value_df(df),
                      self.__get_trend_features_df(df),
                      self.__get_mean_change_and_mean_change_rate(df),
                      self.__get_agg_last_10_sec(df),
                      self.__get_agg_last_15_sec(df),
                      self.__get_agg_last_30_sec(df),
                      self.__get_agg_last_45_sec(df),
                      self.__get_agg_last_60_sec(df),
                      self.__get_agg_last_90_sec(df),
                      self.__get_agg_last_120_sec(df),
                      df_label]

        df = list_of_df[0]
        for df_ in list_of_df[1:]:
            df = pd.merge(df, df_, on=BOOKING_ID, how='left')

        return df
