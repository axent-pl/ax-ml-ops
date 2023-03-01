from re import Pattern
from typing import Union, List
from pandas.core.frame import DataFrame

class ColumnTransformer:
    """
    A class that can be used to apply a transformation to a subset of columns in a pandas DataFrame.

    Attributes
    ----------
    _transformer: callable
        the transformer that should be applied to the selected columns
    _columns: List[Union[str,Pattern]]
        a list of column names or regular expressions that define the subset of columns to be transformed
    _calculated_columns: List[str]
        a list of column names that have been selected for transformation

    Methods
    -------
    __init__(self, transformer, columns:List[Union[str,Pattern]]=[])
        initializes the ColumnTransformer object with the transformer, columns to be transformed
    _calculate_columns(self, X:DataFrame) -> List[str]
        selects the columns to be transformed based on the _columns attribute
    fit(self, X:DataFrame, y=None)
        fits the transformer to the selected columns in the input DataFrame
    transform(self, X:DataFrame, y=None)
        applies the fitted transformer to the selected columns in the input DataFrame
    """

    def __init__(self, transformer, columns:List[Union[str,Pattern]]=[]):
        self._transformer = transformer
        self._columns:List[Union[str,Pattern]] = columns
        self._calculated_columns:List[str] = []
        
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
        self._calculate_columns(X)
        self._transformer.fit(X[self._calculated_columns])
        return self
    
    def transform(self, X:DataFrame, y=None):
        X[self._calculated_columns] = self._transformer.transform(X.loc[:,self._calculated_columns])
        return X