from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from constants import Constants
import pre_processing
import train_test

sc = SparkContext(appName="MLSS")
ssc = StreamingContext(sc, batchDuration=Constants.BATCH_DURATION)
lines = ssc.socketTextStream(hostname=Constants.LOCALHOST, port=Constants.PORT)

lines.flatMap(pre_processing.get_rows_as_dicts) \
    .foreachRDD(train_test.train_models)

ssc.start()
ssc.awaitTermination()
ssc.stop()
