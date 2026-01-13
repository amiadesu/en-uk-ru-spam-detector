from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal, Self
import string

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    A custom text transformer for cleaning string data.

    This transformer is stateless (it does not learn parameters from the data)
    and supports both batch and online learning workflows via `partial_fit`.

    Parameters
    ----------
    mode : Literal['simple', 'no_punct'], default='simple'
        The cleaning strategy to apply:
        - 'simple': Converts the input iterable to a list without modification.
        - 'no_punct': Removes all characters defined in `string.punctuation`.

    Attributes
    ----------
    None
        This estimator is stateless.

    Examples
    --------
    >>> cleaner = TextCleaner(mode='no_punct')
    >>> data = ["Hello, World!", "Hello-World..."]
    >>> cleaner.transform(data)
    ['Hello World', 'HelloWorld']

    # It is compatible with scikit-learn pipelines
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([('cleaner', TextCleaner()), ('model', ...)])
    """
    def __init__(self, mode: Literal['simple', 'no_punct'] = 'simple'):
        self.mode = mode

    def fit(self, X, y = None) -> Self:
        return self
    
    def partial_fit(self, X, y = None) -> Self:
        return self

    def transform(self, X, y = None) -> list[str]:
        if self.mode == 'no_punct':
            translator = str.maketrans("", "", string.punctuation)
            return [text.translate(translator) for text in X]
        return list(X)