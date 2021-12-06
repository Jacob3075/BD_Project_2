import numpy as np
from pyspark import RDD
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.utils import column_or_1d

from constants import Constants
from pre_processing import convert_data_to_df

gaussian = GaussianNB()
bernoulli_classifier = BernoulliNB()
passive_aggressive_classifier = PassiveAggressiveClassifier()
mini_batch_kmeans = MiniBatchKMeans()

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

    _train_test_gaussian(training_x, training_y, testing_x, testing_y)
    _train_test_bernoulli(training_x, training_y, testing_x, testing_y)
    _train_test_passive_aggressive(training_x, training_y, testing_x, testing_y)
    _train_test_mini_batch_kmeans(training_x, training_y, testing_x, testing_y)


def _train_test_gaussian(training_x, training_y, testing_x, testing_y):
    gaussian.partial_fit(training_x, training_y, classes=category_classes)
    predicted_y = gaussian.predict(testing_x)
    _metric_calculation(predicted_y, testing_y, model_name="Gaussian")


def _train_test_passive_aggressive(training_x, training_y, testing_x, testing_y):
    passive_aggressive_classifier.partial_fit(training_x, training_y, classes=category_classes)
    predicted_y = passive_aggressive_classifier.predict(testing_x)
    _metric_calculation(predicted_y, testing_y, model_name="Passive Aggressive")


def _train_test_bernoulli(training_x, training_y, testing_x, testing_y):
    bernoulli_classifier.partial_fit(training_x, training_y, classes=category_classes)
    predicted_y = bernoulli_classifier.predict(testing_x)
    _metric_calculation(predicted_y, testing_y, model_name="Bernoulli")


def _train_test_mini_batch_kmeans(training_x, training_y, testing_x, testing_y):
    mini_batch_kmeans.partial_fit(training_x, training_y)
    predicted_y = mini_batch_kmeans.predict(testing_x)
    _cluster_metric_calculation(predicted_y, testing_y, model_name="Mini Batch Kmeans")


def _cluster_metric_calculation(predicted_y, testing_y, model_name):
    print("----------------------------------------------------------------------------------------------")
    print(model_name)
    print("Accuracy of the model: ", adjusted_rand_score(labels_true=testing_y, labels_pred=predicted_y))
    print("Classification report:")
    metric = np.unique(testing_y)
    print(classification_report(y_true=testing_y, y_pred=predicted_y, labels=metric))


def _metric_calculation(predicted_y, testing_y, model_name):
    print("----------------------------------------------------------------------------------------------")
    print(model_name)
    print("Accuracy of the model: ", accuracy_score(y_true=testing_y, y_pred=predicted_y))
    print("Classification report:")
    metric = np.unique(testing_y)
    print(classification_report(y_true=testing_y, y_pred=predicted_y, labels=metric))
