import json

from pyspark import RDD


def convert_to_df(rdd: RDD):
    """
    :type rdd: pyspark.RDD
    :param rdd:
    """
    # TODO
    val = rdd.collect()
    print("RDD: ", val)


def get_rows_as_dicts(line):
    """
    :type line: str
    :param line: dict with each value as a single row
    :rtype: dict[str, str]
    :return: list of dicts, each dict is a row in the dataset
    """
    return json.loads(line).values()
