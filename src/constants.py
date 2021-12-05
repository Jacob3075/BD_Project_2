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

    KEY_DATES = "Dates"
    KEY_CATEGORY = "Category"
    KEY_DESCRIPT = "Descript"
    KEY_DAY_OF_WEEK = "DayOfWeek"
    KEY_PD_DISTRICT = "PdDistrict"
    KEY_RESOLUTION = "Resolution"
    KEY_ADDRESS = "Address"
    KEY_X = "X"
    KEY_Y = "Y"

    COLUMNS = [
        KEY_DATES,
        KEY_CATEGORY,
        KEY_DESCRIPT,
        KEY_DAY_OF_WEEK,
        KEY_PD_DISTRICT,
        KEY_RESOLUTION,
        KEY_ADDRESS,
        KEY_X,
        KEY_Y,
    ]
