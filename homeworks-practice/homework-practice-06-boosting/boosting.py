from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        n_samples = x.shape[0]
        idx = np.random.choice(n_samples, size=int(self.subsample * n_samples),
                               replace=True)
        x_boot = x[idx]
        y_boot = y[idx]
        old_pred = predictions[idx]
        y_error = y_boot - old_pred
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_boot, y_error)
        new_pred = model.predict(x_boot)

        gamma = self.find_optimal_gamma(y=y_boot,
                                        old_predictions=old_pred,
                                        new_predictions=new_pred)
        self.gammas.append(gamma)
        self.models.append(model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])

        wait = 0
        best_loss = self.loss_fn(y_valid, valid_predictions)
        for _ in range(self.n_estimators):
            self.fit_new_base_model(x=x_train, y=y_train, 
                                    predictions=train_predictions)
            train_predictions = self.predict_proba(x_train)[:, 1]
            valid_predictions = self.predict_proba(x_valid)[:, 1]
            if self.early_stopping_rounds is not None:
                cur_loss = self.loss_fn(y_valid, valid_predictions)
                if cur_loss > best_loss: 
                    best_loss = cur_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.early_stopping_rounds:
                        break
            

        if self.plot:
            self.make_plot(x_valid, y_valid)

    def predict_proba(self, x):
        pred = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            pred = pred + self.learning_rate * gamma * model.predict(x)
        proba = 1 / (1 + np.exp(-pred))
        return np.column_stack([1 - proba, proba])
    

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    def make_plot(self, x, y):
        roc_auc = []
        loss = []
        pred = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            pred = pred + self.learning_rate * gamma * model.predict(x)
            roc_auc.append(roc_auc_score(y == 1, pred))
            loss.append(self.loss_fn(y, pred))
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(roc_auc)
        plt.xlabel('Number of trees')
        plt.ylabel('ROC-AUC')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(loss)
        plt.xlabel('Number of trees')
        plt.ylabel('loss')
        plt.legend()
        

    @property
    def feature_importances_(self):
        importances = np.zeros(self.models[0].feature_importances_.shape[0])
        for model in self.models:
            importances += model.feature_importances_
        mean_importances = importances / len(self.models)
        return mean_importances / sum(mean_importances)
    
