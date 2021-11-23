from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from sklearn import preprocessing as prepro
import json

sc = SparkContext(appName="MLSS")
ssc = StreamingContext(sc, 5)
lines = ssc.socketTextStream("localhost", 6100)


def convert_to_df(rdd):
    # TODO
    print("RDD: ", rdd)


def get_rows_as_dicts(line):
    """
    :param line: dict with each value as a single row
    :return: list of dicts, each dict is a row in the dataset
    """
    return json.loads(line).values()


lines.flatMap(get_rows_as_dicts) \
    .foreachRDD(lambda rdd: convert_to_df(rdd))

ssc.start()
ssc.awaitTermination()

# for i in ['DayOfWeek', 'PdDistrict', 'Address', 'Category']:
#     X[i] = prepro.LabelEncoder().fit_transform(X[i])
#     if i not in ['Category']:
#         test[i] = prepro.LabelEncoder().fit_transform(test[i])
