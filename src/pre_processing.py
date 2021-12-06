import json

from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, hour, month, year

from constants import Constants


def get_rows_as_dicts(line):
    """
    :type line: str
    :param line: dict with each value as a single row
    :rtype: dict[str, str]
    :return: list of dicts, each dict is a row in the dataset
    """
    return json.loads(line).values()


def convert_data_to_df(data):
    """
    :param data:
    :type data: dict
    :return: Input data as dictionary converted to DataFrame
    :rtype: DataFrame
    """
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(data=data, schema=Constants.DATA_SCHEME) \
        .toDF(*Constants.COLUMNS)

    encoding_pairs = [
        Constants.KEY_CATEGORY,
        Constants.KEY_DAY_OF_WEEK,
        Constants.KEY_PD_DISTRICT,
    ]

    for input_name in encoding_pairs:
        df = _label_encoder(df, input_name)

    df = df.withColumn("Timestamp", to_timestamp(df["Dates"]))

    return df.withColumn("Hour", hour(df["Timestamp"])) \
        .withColumn("Month", month(df["Timestamp"])) \
        .withColumn("Year", year(df["Timestamp"]))


def _label_encoder(data_frame, input_column_name):
    """
    Replaces the input column with encoded values
    :param data_frame: Dataframe to modify
    :type data_frame: DataFrame
    :param input_column_name: Name of the column to encode
    :type input_column_name: str
    :return: Updated dataframe
    :rtype: DataFrame
    """
    return StringIndexer(inputCol=input_column_name, outputCol=f"{input_column_name}_encoded") \
        .fit(data_frame) \
        .transform(data_frame) \
        .drop(input_column_name) \
        .withColumnRenamed(existing=f"{input_column_name}_encoded", new=input_column_name)
