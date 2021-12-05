from pyspark.sql.types import StructType, StructField, StringType, DoubleType


class Constants:
    PORT = 6100
    LOCALHOST = "localhost"
    BATCH_DURATION = 5

    DATA_SCHEME = StructType(
        [
            StructField("feature0", StringType()),
            StructField("feature1", StringType()),
            StructField("feature2", StringType()),
            StructField("feature3", StringType()),
            StructField("feature4", StringType()),
            StructField("feature5", StringType()),
            StructField("feature6", StringType()),
            StructField("feature7", DoubleType()),
            StructField("feature8", DoubleType())
        ]
    )
