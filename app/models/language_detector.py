from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array

class LanguageDetector(BaseEstimator, ClassifierMixin):
    """
    A custom language detector model wrapper.

    This wrapper is designed to automatically encode and decode target variables.

    Parameters
    ----------
    clf : ClassifierMixin | Pipeline
        The classifier or pipeline with model that will perform all predictions.
    """
    def __init__(self, clf: ClassifierMixin | Pipeline):
        self.clf = clf
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True, ensure_2d=False, dtype=None  )

        self.encoder_ = LabelEncoder()

        y_encoded = self.encoder_.fit_transform(y)
        self.classes_ = self.encoder_.classes_

        self.clf_: ClassifierMixin | Pipeline = clone(self.clf)
        self.clf_.fit(X, y_encoded)

        return self

    def predict(self, X):
        check_is_fitted(self, ['clf_', 'encoder_'])

        X = check_array(X, ensure_2d=False, dtype=None)
        
        y_pred = self.clf_.predict(X)
        return self.encoder_.inverse_transform(y_pred)

    def predict_proba(self, X):
        check_is_fitted(self, 'clf_')

        return self.clf_.predict_proba(X)
    
    def decision_function(self, X):
        check_is_fitted(self, 'clf_')

        return self.clf_.decision_function(X)