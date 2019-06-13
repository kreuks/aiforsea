import pyspark.sql.functions as F
from pyspark.ml import Transformer
from pyspark.sql import DataFrame, Window

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
LABEL = 'label'


class FeatureEngineeringTransformer(Transformer):
    def _get_last_n_sec(self, df: DataFrame, n: int):
        window = Window.partitionBy(BOOKING_ID).orderBy(F.col(SECOND).desc())
        df = df.select('*', F.rank().over(window).alias('rank'))
        df = df.filter(F.col('rank') <= n).drop('rank')

        _aggregate_column = [F.avg(ACCURACY).alias('avg_accuracy_last_{}_sec'.format(n)),
                             F.avg(BEARING).alias('avg_bearing_last_{}_sec'.format(n)),
                             F.avg(ACC_X).alias('avg_acc_x_last_{}_sec'.format(n)),
                             F.avg(ACC_Y).alias('avg_acc_y_last_{}_sec'.format(n)),
                             F.avg(ACC_Z).alias('avg_acc_z_last_{}_sec'.format(n)),
                             F.avg(GYRO_X).alias('avg_gyro_x_last_{}_sec'.format(n)),
                             F.avg(GYRO_Y).alias('avg_gyro_y_last_{}_sec'.format(n)),
                             F.avg(GYRO_Z).alias('avg_gyro_z_last_{}_sec'.format(n)),
                             F.avg(SPEED).alias('avg_speed_last_{}_sec'.format(n))]

        df = df.groupBy(BOOKING_ID).agg(*_aggregate_column)

        return df

    def _get_last_5_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 5)

    def _get_last_15_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 15)

    def _get_last_30_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 30)

    def _get_last_45_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 45)

    def _get_last_60_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 60)

    def _get_last_75_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 75)

    def _get_last_90_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 90)

    def _get_last_105_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 105)

    def _get_last_120_sec(self, df: DataFrame):
        return self._get_last_n_sec(df, 120)

    def _get_time_series_df(self, df: DataFrame):
        last_5_sec_df = self._get_last_5_sec(df)
        last_15_sec_df = self._get_last_15_sec(df)
        last_30_sec_df = self._get_last_30_sec(df)
        last_45_sec_df = self._get_last_45_sec(df)
        last_60_sec_df = self._get_last_60_sec(df)
        last_75_sec_df = self._get_last_75_sec(df)
        last_90_sec_df = self._get_last_90_sec(df)
        last_105_sec_df = self._get_last_105_sec(df)
        last_120_sec_df = self._get_last_120_sec(df)
        _df_list = [last_5_sec_df, last_15_sec_df, last_30_sec_df, last_45_sec_df, last_60_sec_df, last_75_sec_df,
                    last_90_sec_df, last_105_sec_df, last_120_sec_df]

        _df = _df_list[0]
        for x in _df_list[1:]:
            _df = _df.join(x, on=BOOKING_ID, how='left')
        return _df

    def _transform(self, df: DataFrame) -> DataFrame:
        window_booking_id = Window.partitionBy(BOOKING_ID).orderBy(F.col(SECOND).desc())
        groupby_column = [BOOKING_ID]

        aggregate_column = [F.max(ACCURACY).alias('max_accuracy'),
                            F.max(BEARING).alias('max_bearing'),
                            F.max(ACC_X).alias('max_acc_x'),
                            F.max(ACC_Y).alias('max_acc_y'),
                            F.max(ACC_Z).alias('max_acc_z'),
                            F.max(GYRO_X).alias('max_gyro_x'),
                            F.max(GYRO_Y).alias('max_gyro_y'),
                            F.max(GYRO_Z).alias('max_gyro_z'),
                            F.max(SECOND).alias('max_second'),
                            F.max(SPEED).alias('max_speed'),

                            F.min(ACCURACY).alias('min_accuracy'),
                            F.min(BEARING).alias('min_bearing'),
                            F.min(ACC_X).alias('min_acc_x'),
                            F.min(ACC_Y).alias('min_acc_y'),
                            F.min(ACC_Z).alias('min_acc_z'),
                            F.min(GYRO_X).alias('min_gyro_x'),
                            F.min(GYRO_Y).alias('min_gyro_y'),
                            F.min(GYRO_Z).alias('min_gyro_z'),
                            F.min(SECOND).alias('min_seconds'),
                            F.min(SPEED).alias('min_speed'),

                            F.avg(ACCURACY).alias('avg_accuracy'),
                            F.avg(BEARING).alias('avg_bearing'),
                            F.avg(ACC_X).alias('avg_acc_x'),
                            F.avg(ACC_Y).alias('avg_acc_y'),
                            F.avg(ACC_Z).alias('avg_acc_z'),
                            F.avg(GYRO_X).alias('avg_gyro_x'),
                            F.avg(GYRO_Y).alias('avg_gyro_y'),
                            F.avg(GYRO_Z).alias('avg_gyro_z'),
                            F.avg(SECOND).alias('avg_second'),
                            F.avg(SPEED).alias('avg_speed'),

                            F.sum(ACCURACY).alias('sum_accuracy'),
                            F.sum(BEARING).alias('sum_bearing'),
                            F.sum(ACC_X).alias('sum_acc_x'),
                            F.sum(ACC_Y).alias('sum_acc_y'),
                            F.sum(ACC_Z).alias('sum_acc_z'),
                            F.sum(GYRO_X).alias('sum_gyro_x'),
                            F.sum(GYRO_Y).alias('sum_gyro_y'),
                            F.sum(GYRO_Z).alias('sum_gyro_z'),
                            F.sum(SECOND).alias('sum_second'),
                            F.sum(SPEED).alias('sum_speed'),

                            F.stddev(ACCURACY).alias('stddev_accuracy'),
                            F.stddev(BEARING).alias('stddev_bearing'),
                            F.stddev(ACC_X).alias('stddev_acc_x'),
                            F.stddev(ACC_Y).alias('stddev_acc_y'),
                            F.stddev(ACC_Z).alias('stddev_acc_z'),
                            F.stddev(GYRO_X).alias('stddev_gyro_x'),
                            F.stddev(GYRO_Y).alias('stddev_gyro_y'),
                            F.stddev(GYRO_Z).alias('stddev_gyro_z'),
                            F.stddev(SECOND).alias('stddev_second'),
                            F.stddev(SPEED).alias('stddev_speed'),

                            # TODO implement mean absolute deviation

                            ]

        df_time_series_features = self._get_time_series_df(df)
        df = df.groupBy(*groupby_column) \
            .agg(*aggregate_column)

        df = df.join(df_time_series_features, on=BOOKING_ID)

        return df
