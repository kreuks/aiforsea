from pyspark.sql import SparkSession
import pandas as pd
from safety.feature_engineering import FeatureEngineeringTransformer
from safety.feature_engineering_pandas import FeatureEngineering

if __name__ == '__main__':
    # spark = SparkSession.builder.config('spark.driver.memory', '5g').getOrCreate()
    #
    # df = spark.read.parquet('data.parquet')
    # fet = FeatureEngineeringTransformer()
    # df = fet.transform(df)
    # df.write.parquet('data_clean.parquet')

    df = pd.read_csv('safety/data/data.csv/part-00000-1e8e58a1-ca88-449f-bd9f-e042ec813ea5-c000.csv')
    # df = pd.read_csv('safety/data/data.csv/test.csv')
    print('finish load')
    print('starting fe')
    fe = FeatureEngineering()
    df = fe.transform(df)
    df.to_csv('coba-final.csv', header=True, index=False)
