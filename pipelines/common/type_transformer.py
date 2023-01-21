from pandas.core.frame import DataFrame

class TypeTransformer:

    def __init__(self, dtype: str):
        self._dtype = dtype
        
    def fit(self, X:DataFrame, y=None):
        return self
    
    def transform(self, X:DataFrame, y=None):
        for c in X.columns:
            X[c] = X[c].astype(self._dtype)
        return X