import json

import numpy as np
from pyspark import RDD
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, hour, month, year
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d

from constants import Constants

sgd_classifier = SGDClassifier()
category_classes = [float(x) for x in list(range(Constants.CATEGORY_LEN))]


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

    encoding_pairs = {
        Constants.KEY_CATEGORY: "Category_encoded",
        Constants.KEY_DAY_OF_WEEK: "DayOfWeek_encoded",
        Constants.KEY_PD_DISTRICT: "PdDistrict_encoded"
    }

    for input_name, output_name in encoding_pairs.items():
        df = label_encoder(df, input_name, output_name)
        # TODO: RENAME COLUMNS BACK TO ORIGINAL

    df = df.drop(*list(encoding_pairs.keys())) \
        .withColumn("Timestamp", to_timestamp(df["Dates"]))

    df = df.withColumn("Hour", hour(df["Timestamp"])) \
        .withColumn("Month", month(df["Timestamp"])) \
        .withColumn("Year", year(df["Timestamp"]))

    # df.show()
    return df


def metric_calculation(predicted_y, testing_y):
    metric = np.unique(testing_y)
    print(metric)
    print("Accuracy of the model: ", accuracy_score(predicted_y, testing_y))
    print("Classification report:")
    print(classification_report(y_true=testing_y, y_pred=predicted_y, labels=metric))


def train_model(df):
    """
    :param df:
    :type df: DataFrame
    """
    x_columns = [
        Constants.KEY_X,
        Constants.KEY_Y,
        f"{Constants.KEY_DAY_OF_WEEK}_encoded",
        f"{Constants.KEY_PD_DISTRICT}_encoded",
        "Hour",
        "Month",
        "Year",
    ]
    coord_x = np.asarray(df.select(x_columns).collect())
    coord_y = np.asarray(df.select(f"{Constants.KEY_CATEGORY}_encoded").collect())

    coord_y = column_or_1d(coord_y, warn=True)

    training_x, testing_x, training_y, testing_y = train_test_split(coord_x, coord_y, test_size=0.2, random_state=0)

    # classifier used here
    sgd_classifier.partial_fit(training_x, training_y, classes=category_classes)

    # testing starts here
    predicted_y_global = sgd_classifier.predict(testing_x)

    # Metrics is calculated here
    metric_calculation(predicted_y_global, testing_y)


def process_data(rdd):
    """
    :param rdd:
    :type rdd: RDD
    """
    if rdd.isEmpty():
        return

    df = convert_data_to_df(rdd.collect())
    train_model(df)


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
