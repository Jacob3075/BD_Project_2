from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from constants import Constants
from pre_processing import get_rows_as_dicts
from train_test import train_models

sc = SparkContext(appName="MLSS")
ssc = StreamingContext(sc, batchDuration=Constants.BATCH_DURATION)
lines = ssc.socketTextStream(hostname=Constants.LOCALHOST, port=Constants.PORT)

lines.flatMap(get_rows_as_dicts) \
    .foreachRDD(train_models)

ssc.start()
ssc.awaitTermination()
