from re import Pattern
from typing import Union, List
from pandas.core.frame import DataFrame

class ColumnTransformer:
    
    def __init__(self, transformer, columns:List[Union[str,Pattern]]=[], fit_once=True):
        self._fitted = False
        self._fit_once = False
        self._transformer = transformer
        self._columns = columns
        self._calculated_columns = []
        
    def _calculate_columns(self, X:DataFrame) -> List[str]:
        _columns = []
        for col_def in self._columns:
            if isinstance(col_def,Pattern):
                for col in X.columns:
                    if col_def.match(col):
                        _columns.append(col)
            else:
                _columns.append(col_def)
        self._calculated_columns = _columns
        return _columns
        
    def fit(self, X:DataFrame, y=None):
        if self._fit_once and not self._fitted or not self._fit_once:
            self._calculate_columns(X)
            self._transformer.fit(X[self._calculated_columns])
            self._fitted = True
        return self
    
    def transform(self, X:DataFrame, y=None):
        X[self._calculated_columns] = self._transformer.transform(X.loc[:,self._calculated_columns])
        return X