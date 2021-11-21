from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext(appName="NetworkWordCount")
ssc = StreamingContext(sc, 5)
lines = ssc.socketTextStream("localhost", 6100)

lines.pprint()

ssc.start()
ssc.awaitTermination()
