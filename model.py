from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
import os

class ClassificationModel():
    def __init__(self, type='logistic-regression'):
        self.model = self.get_model(type)
        self.model_type = type
        self.input_dim = 0

    def get_model(self, type):
        if type == 'LogisticRegression':
            return LogisticRegression(solver='liblinear')
        elif type == 'Multinomial':
            return LogisticRegression(multi_class='multinomial', solver="lbfgs")
        else:
            raise Exception('Invalid model type')

    def fit(self, dataset, test_size=0.15, seed=1337):
        X = dataset.embeds
        y = dataset.ratings

        self.input_dim = X.shape[1]
        self.model.fit(X, y)

        return self._results(X, y)

    def test(self, dataset):
        X = dataset.embeds
        y = dataset.ratings
        return self._results(X, y)

    def _results(self, X_test, y_test):
        y_pred_class = self.model.predict(X_test)

        accuracy_score = metrics.accuracy_score(y_test, y_pred_class)
        average = 'weighted' if self.model_type == 'Multinomial' else 'binary'

        f1_score = metrics.f1_score(y_test, y_pred_class, average=average)

        return {
            'accuracy': accuracy_score,
            'f1': f1_score
        }

    def save(self, path):
        # make directory within path if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        weights = self.model.coef_
        biases = self.model.intercept_
        classes = self.model.classes_
        np.savez(path, weights=weights, biases=biases, classes=classes)

    def load(self, path):
        # check if path exists
        if not os.path.exists(path):
            return False

        data = np.load(path)
        weights = data['weights']
        biases = data['biases']
        classes = data['classes']

        model = self.get_model(type)
        model.coef_ = np.array(weights)
        model.intercept_ = np.array(biases)
        model.classes_ = classes

        self.model = model