from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from constants import Constants
from utils import convert_to_df, get_rows_as_dicts

sc = SparkContext(appName="MLSS")
ssc = StreamingContext(sc, batchDuration=Constants.BATCH_DURATION)
lines = ssc.socketTextStream(hostname=Constants.LOCALHOST, port=Constants.PORT)

lines.flatMap(get_rows_as_dicts) \
    .foreachRDD(convert_to_df)

ssc.start()
ssc.awaitTermination()

# for i in ['DayOfWeek', 'PdDistrict', 'Address', 'Category']:
#     X[i] = prepro.LabelEncoder().fit_transform(X[i])
#     if i not in ['Category']:
#         test[i] = prepro.LabelEncoder().fit_transform(test[i])
