from sklearn.base import BaseEstimator, TransformerMixin

class LowerCaseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.applymap(lambda s: s.lower() if isinstance(s, str) else s)
