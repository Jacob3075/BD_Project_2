import numpy as np
from pyspark import RDD
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d

from constants import Constants
from pre_processing import convert_data_to_df

sgd_classifier = SGDClassifier()
passive_aggressive_classifier = PassiveAggressiveClassifier()

category_classes = [float(x) for x in list(range(Constants.CATEGORY_LEN))]


def train_models(rdd):
    """
    :param rdd:
    :type rdd: RDD
    """
    if rdd.isEmpty():
        return

    df = convert_data_to_df(rdd.collect())

    x_columns = [
        Constants.KEY_X,
        Constants.KEY_Y,
        Constants.KEY_DAY_OF_WEEK,
        Constants.KEY_PD_DISTRICT,
        "Hour",
        "Month",
        "Year",
    ]
    coord_x = np.asarray(df.select(x_columns).collect())
    coord_y = np.asarray(df.select(Constants.KEY_CATEGORY).collect())

    coord_y = column_or_1d(coord_y, warn=True)

    training_x, testing_x, training_y, testing_y = train_test_split(coord_x, coord_y, test_size=0.2, random_state=0)

    _train_test_sgd(training_x, training_y, testing_x, testing_y)
    _train_test_passive_aggressive(training_x, training_y, testing_x, testing_y)


def _train_test_sgd(training_x, training_y, testing_x, testing_y):
    sgd_classifier.partial_fit(training_x, training_y, classes=category_classes)
    predicted_y_global = sgd_classifier.predict(testing_x)
    _metric_calculation(predicted_y_global, testing_y, model_name="SGD")


def _train_test_passive_aggressive(training_x, training_y, testing_x, testing_y):
    passive_aggressive_classifier.partial_fit(training_x, training_y, classes=category_classes)
    predicted_y_global = passive_aggressive_classifier.predict(testing_x)
    _metric_calculation(predicted_y_global, testing_y, model_name="Passive Aggressive")


def _metric_calculation(predicted_y, testing_y, model_name):
    print("----------------------------------------------------------------------------------------------")
    print(model_name)
    print("Accuracy of the model: ", accuracy_score(predicted_y, testing_y))
    print("Classification report:")
    metric = np.unique(testing_y)
    print(classification_report(y_true=testing_y, y_pred=predicted_y, labels=metric))
