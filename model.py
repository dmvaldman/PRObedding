import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os

class ClassificationModel():
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')
        self.input_dim = 0

    def fit(self, dataset, test_size=0.15, seed=1337):
        X = dataset.embeds
        y = dataset.ratings

        # apply random shuffle to both ratings and embeddings
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        self.input_dim = X_train.shape[1]
        self.model.fit(X_train, y_train)

        return self._results(X_test, y_test)

    def _results(self, X_test, y_test):
        # make class predictions for X_test_dtm
        y_pred_class = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]

        accuracy_score = metrics.accuracy_score(y_test, y_pred_class)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred_class)
        roc_score = metrics.roc_auc_score(y_test, y_pred_prob)

        return {
            'accuracy_score': accuracy_score,
            'confusion_matrix': confusion_matrix,
            'roc_score': roc_score
        }

    def save(self, type):
        path = f'models/{type}/model.npz'

        # make directory if it doesn't exist
        if not os.path.exists(f'models/{type}'):
            os.makedirs(f'models/{type}')

        weights = self.model.coef_
        biases = self.model.intercept_
        classes = self.model.classes_
        np.savez(path, weights=weights, biases=biases, classes=classes)

    def load(self, type):
        path = f'models/{type}/model.npz'

        # check if path exists
        if not os.path.exists(path):
            return False

        data = np.load(path)
        weights = data['weights']
        biases = data['biases']
        classes = data['classes']

        model = LogisticRegression(solver='liblinear')
        model.coef_ = np.array(weights)
        model.intercept_ = np.array(biases)
        model.classes_ = classes

        self.model = model