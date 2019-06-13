import argparse
import glob

import pandas as pd

from safety.feature_engineering import FeatureEngineering, BOOKING_ID, LABEL
from safety.modeler import XGBoostModeler, Optimizer
from safety.util import logger, read_multiple_csv_pandas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--mode', type=str,
        help='Either train or predict',
        required=True
    )
    parser.add_argument(
        '-if', '--input_features_folder', type=str,
        help='Input folder of your features csv data.',
        required=True
    )
    parser.add_argument(
        '-il', '--input_label_folder', type=str,
        help='Input folder of your label csv data.',
        required=False
    )
    parser.add_argument(
        '-o', '--output_path', type=str,
        help='Output path for prediction result.',
        required=True
    )

    args = parser.parse_args()
    mode = args.mode
    input_features_folder = args.input_features_folder
    input_label_folder = args.input_label_folder
    output_path = args.output_path

    if mode == 'training':
        logger.info('start load dataframe')

        df_features = read_multiple_csv_pandas(input_features_folder)
        df_label = read_multiple_csv_pandas(input_label_folder)
        df_label = df_label.groupby(BOOKING_ID).agg({LABEL: 'last'}).reset_index()

        logger.info('finished load dataframe')
        logger.info('start feature engineering')

        fe = FeatureEngineering()
        df_features = fe.transform(df_features)

        logger.info('finished feature engineering')
        logger.info('start modeling')

        df = pd.merge(df_features, df_label, on=BOOKING_ID, how='left')
        df_features = df[BOOKING_ID]
        df_label = df[LABEL]

        opt = Optimizer(df_features, df_label)
        opt.run()

        logger.info('finish modeling')

    elif mode == 'predict':
        """
        Your csv file should contain bookingID column.
        """

        logger.info('start load dataframe')

        df = read_multiple_csv_pandas(input_features_folder)

        logger.info('finished load dataframe')
        logger.info('start feature engineering')

        fe = FeatureEngineering()
        df = fe.transform(df)

        logger.info('finished feature engineering')
        logger.info('start predict')

        booking_ids = df[BOOKING_ID].values
        df = df.drop(BOOKING_ID, axis=1)

        model_path = sorted(glob.glob('safety/models/xgboost/*/*.model'))[-1]

        xgb_modeler = XGBoostModeler()
        xgb_modeler.load(model_path)
        logger.info('loaded model is {}'.format(model_path.split('/')[-2]))
        result = xgb_modeler.predict(df)

        logger.info('finished predict')

        df = pd.DataFrame({BOOKING_ID: booking_ids, 'probability': result})
        df.to_csv(output_path, header=True, index=False)
