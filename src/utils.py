import json

from pyspark import RDD
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, hour, month, year

from constants import Constants


def convert_to_df(spark, rdd):
    """
    :param spark: SparkSession needed to create dataframe
    :type spark: SparkSession
    :param rdd:
    :type rdd: RDD
    """
    if rdd.isEmpty():
        return

    df = spark.createDataFrame(data=rdd.collect(), schema=Constants.DATA_SCHEME)
    df = df.toDF(*Constants.COLUMNS)
    encoding_pairs = {
        Constants.KEY_CATEGORY: "Category_encoded",
        Constants.KEY_DAY_OF_WEEK: "DayOfWeek_encoded",
        Constants.KEY_PD_DISTRICT: "PdDistrict_encoded"
    }

    for input_name, output_name in encoding_pairs.items():
        df = label_encoder(df, input_name, output_name)

    df = df.drop(*list(encoding_pairs.keys()))

    df = df.withColumn("Timestamp", to_timestamp(df["Dates"]))

    df = df.withColumn("Hour", hour(df["Timestamp"])) \
        .withColumn("Month", month(df["Timestamp"])) \
        .withColumn("Year", year(df["Timestamp"]))

    df.show()


def get_rows_as_dicts(line):
    """
    :type line: str
    :param line: dict with each value as a single row
    :rtype: dict[str, str]
    :return: list of dicts, each dict is a row in the dataset
    """
    return json.loads(line).values()


def label_encoder(data_frame, input_column_name, output_column_name):
    """
    :param data_frame: Dataframe to modify
    :type data_frame: DataFrame
    :param input_column_name: Name of the column to encode
    :type input_column_name: str
    :param output_column_name: Encoded column name
    :type output_column_name: str
    :return: Updated dataframe
    :rtype: DataFrame
    """
    return StringIndexer(inputCol=input_column_name, outputCol=output_column_name) \
        .fit(data_frame) \
        .transform(data_frame)
