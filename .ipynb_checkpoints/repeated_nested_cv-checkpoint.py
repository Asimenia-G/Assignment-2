import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, fbeta_score, balanced_accuracy_score, precision_score,
    recall_score, matthews_corrcoef, roc_auc_score
)
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")


class RepeatedNestedCV:
    def __init__(self, estimators, param_grids, R=10, N=5, K=3, random_state=42, scoring='f1'):
        self.estimators = estimators
        self.param_grids = param_grids
        self.R = R
        self.N = N
        self.K = K
        self.random_state = random_state
        self.scoring = scoring
        self.results = []

    def evaluate_metrics(self, y_true, y_pred, y_prob=None):
        return {
            'F1': f1_score(y_true, y_pred),
            'F2': fbeta_score(y_true, y_pred, beta=2),
            'BA': balanced_accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'AUC': roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
        }

    def run(self, X, y):
        for r in range(self.R):
            outer_cv = StratifiedKFold(n_splits=self.N, shuffle=True, random_state=self.random_state + r)
            for outer_train_idx, outer_test_idx in outer_cv.split(X, y):
                X_train, X_test = X[outer_train_idx], X[outer_test_idx]
                y_train, y_test = y[outer_train_idx], y[outer_test_idx]

                for est_name, estimator in self.estimators.items():
                    grid = GridSearchCV(
                        estimator,
                        self.param_grids[est_name],
                        scoring=self.scoring,
                        cv=StratifiedKFold(n_splits=self.K, shuffle=True, random_state=self.random_state + r),
                        n_jobs=-1,
                        refit=True
                    )
                    grid.fit(X_train, y_train)

                    model = clone(grid.best_estimator_)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                    except:
                        y_prob = None

                    metrics = self.evaluate_metrics(y_test, y_pred, y_prob)
                    metrics['estimator'] = est_name
                    metrics['round'] = r
                    self.results.append(metrics)

        return pd.DataFrame(self.results)
