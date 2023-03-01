from itertools import combinations
from typing import List
from pandas.core.frame import DataFrame

class FillnaAssociationTransformer:

    def __init__(self, min_support:int = 2, min_x_dim:int = 1, verbose:int = 0):
        self._min_support:int = min_support
        self._min_x_dim:int = min_x_dim
        self._verbose:int = verbose

    def _fit_transform(self, df:DataFrame, x_columns:List[str], y_column:str) -> DataFrame:
        cols = x_columns
        col = y_column

        # extracting rules with min_support
        df_rules = df[cols+[col]]\
            .dropna(subset=list(cols), how='any')\
            .groupby(cols)[[col]]\
            .agg(nunique=(col,'nunique'),support=(col,'count'),val=(col,'last'))\
            .reset_index()\
            .query(f'nunique == 1 and support >= {self._min_support}')\
            .copy()
        df_rules.rename(columns={'val':f'{col}_val'}, inplace=True)
        df_rules.drop(['nunique','support'], axis=1, inplace=True)

        # check if can be applied than apply
        to_be_filled_count = df[df[col].isna()]\
            .merge(df_rules, how='inner', on=cols)\
            .dropna(subset=list(cols), how='any')\
            .shape[0]

        # apply rules
        if to_be_filled_count > 0:
            df_merged = df.merge(df_rules, how='outer', on=cols)
            df[f'{col}_val'] = df_merged[f'{col}_val']
            df.loc[(df[col].isna()) & (df[f'{col}_val'].notna()), col] = df[f'{col}_val']
            df.drop(f'{col}_val', axis=1, inplace=True)
            if self._verbose > 0:
                print(cols, col, to_be_filled_count)
        
        return df

    def fit(self, X:DataFrame, y=None):
        return self
    
    def transform(self, df:DataFrame, y=None):
        columns = list(df.columns)
        for i in range(self._min_x_dim,len(columns)):
            for combination in combinations(columns, i):
                x_columns = list(combination)
                for y_column in columns:
                    if y_column not in x_columns:
                        df = self._fit_transform(df, x_columns, y_column)
        return df