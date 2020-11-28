from sklearn import metrics, model_selection
import tensorflow as ts
from tensorflow.contrib import learn


def main(unused_args):
    # load dataset
    iris = learn.datasets.load_dataset('iris')
    x_train, x_test, _y_train, _y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2,
                                                                          random_state=42)

    # Build 3 Layer DNN with 10, 20, 10 units respectively
    classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

    # fit and predict
    classifier.fit(x_train, _y_train, steps=200)
    score = metrics.accuracy_score(_y_test, classifier.predict(_y_test))
    print('Accuracy:  {0:f}'.format(score))
