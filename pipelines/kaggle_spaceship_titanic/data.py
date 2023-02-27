import os
import re
from io import StringIO
import boto3
import numpy as np
from numpy import ndarray
import pandas as pd
from itertools import combinations

from typing import List
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from .common.transformer.column_transformer import ColumnTransformer
from .common.transformer.type_transformer import TypeTransformer
from .common_data import TrainTestDataProvider

class PipelineUtils:

    def add_dummies(df, columns: List[str], drop_first=True, pass_nan=True):
        new_columns = []
        for c in columns:
            dummies = pd.get_dummies(df[c], prefix=f'cat_{c}', drop_first=drop_first)
            new_columns += list(dummies.columns)
            df[list(dummies.columns)] = dummies
            if pass_nan:
                df.loc[(df[c].isnull()),list(dummies.columns)] = np.nan
        return df

    def del_dummies(df, columns: List[str]):
        for c in columns:
            dummies = pd.get_dummies(df[c], prefix=f'cat_{c}', drop_first=False)
            new_columns += list(dummies.columns)
            df.drop(new_columns, axis=1, inplace=True)
        return df

class CustomPipelineUtils:
    
    def add_columns(df):
        df[['PassengerId_Group','PassengerId_Number']] = df['PassengerId'].str.split('_', expand=True)
        df[['Cabin_Deck','Cabin_Num','Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
        df[['Name_First','Name_Last']] = df['Name'].str.split(' ', expand=True)

        exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df['Spending'] = (df[exp_cols].max(axis=1) > 0)
        df['Spending'] = df['Spending'].astype(float)
        for c in exp_cols:
            df.loc[(df[c].isna()) & (df['Spending'] == 0), 'Spending'] = np.nan

        return df

    def fillna_data_mining(df, x_columns, y_column, min_support):
        cols = x_columns
        col = y_column

        # extracting rules with min_support
        df_rules = df[cols+[col]]\
            .dropna(subset=list(cols), how='any')\
            .groupby(cols)[[col]]\
            .agg(nunique=(col,'nunique'),support=(col,'count'),val=(col,'last'))\
            .reset_index()\
            .query(f'nunique == 1 and support >= {min_support}')\
            .copy()
        df_rules.rename(columns={'val':f'{col}_val'}, inplace=True)
        df_rules.drop(['nunique','support'], axis=1, inplace=True)

        # if rules can be applied than apply
        to_be_filled_count = df[df[col].isna()]\
            .merge(df_rules, how='inner', on=cols)\
            .dropna(subset=list(cols), how='any')\
            .shape[0]

        if to_be_filled_count > 0:
            df_merged = df.merge(df_rules, how='outer', on=cols)
            df[f'{col}_val'] = df_merged[f'{col}_val']
            df.loc[(df[col].isna()) & (df[f'{col}_val'].notna()), col] = df[f'{col}_val']
            df.drop(f'{col}_val', axis=1, inplace=True)
            print(cols, col, to_be_filled_count)
        
        return df

    def fillna_rules(df):
        all_x_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Cabin_Deck', 'Cabin_Num', 'Cabin_Side', 'Spending', 'PassengerId_Group', 'Name_Last']
        all_y_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'Cabin_Deck', 'Cabin_Num', 'Cabin_Side', 'Spending']
        min_support = 2

        for i in range(len(all_x_columns)-1,3,-1):
            for comination in combinations(all_x_columns, i):
                x_columns = list(comination)
                for y_column in all_y_columns:
                    if y_column not in x_columns:
                        df = CustomPipelineUtils.fillna_data_mining(df, x_columns, y_column, min_support)

        for c in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
            df.loc[(df[c].isna()) & (df['Spending'] == 0.0), c] = 0.0

        return df

    def fillna_imputer(df):
        cat_cols = ['HomePlanet','Destination','Cabin_Deck','Cabin_Side']
        num_cols = ['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Cabin_Num','PassengerId_Group','PassengerId_Number','CryoSleep','VIP']

        for c in cat_cols:
            dummies = pd.get_dummies(df[c], prefix=f'dummy_{c}', drop_first=False)
            df[list(dummies.columns)] = dummies
            df.loc[(df[c].isnull()),list(dummies.columns)] = np.nan
            
        cols = []
        cols += num_cols
        cols += [ c for c in df.columns if c.startswith('dummy_') ]

        imputer = KNNImputer()
        df[cols] = imputer.fit_transform(df[cols])

        for c in cat_cols:
            column_prefix = f'dummy_{c}_'
            dummy_cols = [ dc for dc in df.columns if dc.startswith(column_prefix) ]
            df.loc[df[c].isna(), c] = df[dummy_cols].idxmax(axis=1).str.replace(column_prefix, '')
            df.drop(dummy_cols, axis=1, inplace=True)

        return df

    def fillna_fix_cabin_num(df):
        df['Cabin_Deck_Side'] = df['Cabin_Deck'].astype('str') + "/" + df['Cabin_Side'].astype('str')
        for cds in df['Cabin_Deck_Side'].unique():
            df_slice = df[(df['Cabin'].notna()) & (df['Cabin_Deck_Side']==cds)]
            a,b = np.polyfit(df_slice['PassengerId_Group'], df_slice['Cabin_Num'], deg=1)
            df.loc[(df['Cabin_Deck_Side']==cds) & (df['Cabin'].isna()), 'Cabin_Num'] = a * df['PassengerId_Group'] + b
        return df

    def log_scale(df):
        cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for c in cols:
            df[c] = np.log(df[c]+1.0)
        return df

    def _bin_column(df, column, bins, right=False):
        df[f'bin_{column}'] = pd.cut(df[column],bins=bins, right=right)
        dummies = pd.get_dummies(df[f'bin_{column}'], prefix=f'cat_bin_{column}')
        df[list(dummies.columns)[1:]] = dummies[list(dummies.columns)[1:]]
        return df

    def bin_columns(df):
        # df = CustomPipelineUtils._bin_column(df, 'Age', [0,13,18,26,29,49,200])
        # df = CustomPipelineUtils._bin_column(df, 'Cabin_Num', list(range(0,2400,300)))
        # df = CustomPipelineUtils._bin_column(df, 'PassengerId_Group', list(range(0,11000,1000)))
        return df
    
    def add_summary_columns(df):
        df['agr_Count_Cabin_Num'] = df.groupby(['Cabin_Num']).transform('count')['PassengerId'].fillna(0.0)
        df['agr_Sum_Expenses'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
        df['agr_Count_PassengerId_Group'] = df.groupby(['PassengerId_Group']).transform('count')['PassengerId'].fillna(0.0)
        df['agr_Count_Name_Last'] = df.groupby(['Name_Last']).transform('count')['PassengerId'].fillna(0.0)

        df['agr_Age_Cabin_Num'] = df['Age'] * df['Cabin_Num']
        df['agr_Age_Expenses'] = df['Age'] * df['agr_Sum_Expenses']

        return df

cat_cols = ['HomePlanet','Destination','Cabin_Deck','Cabin_Side']
num_cols = ['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Cabin_Num','PassengerId_Group','PassengerId_Number','CryoSleep','VIP']

def load_and_transform_data():
    df_train = pd.read_csv(os.path.dirname(__file__)+'/train.csv')
    df_train['train'] = 1

    df_test = pd.read_csv(os.path.dirname(__file__)+'/test.csv')
    df_test['train'] = 0

    df = pd.concat([df_test, df_train], ignore_index=True)

    pipeline = Pipeline(steps=[
        ('split', FunctionTransformer(CustomPipelineUtils.add_columns)),
        ('astype_num', ColumnTransformer(TypeTransformer('float64'), columns=num_cols)),
        ('fillna_rules', FunctionTransformer(CustomPipelineUtils.fillna_rules)),
        ('fillna_imputer', FunctionTransformer(CustomPipelineUtils.fillna_imputer)),
        ('fillna_cabin_num', FunctionTransformer(CustomPipelineUtils.fillna_fix_cabin_num)),
        ('log_scale', FunctionTransformer(CustomPipelineUtils.log_scale)),
        ('astype_cat', ColumnTransformer(TypeTransformer('category'), columns=cat_cols)),
        ('add_dummies', FunctionTransformer(PipelineUtils.add_dummies, kw_args={'columns':cat_cols})),
        ('add_summary_columns', FunctionTransformer(CustomPipelineUtils.add_summary_columns)),
        ('bin_columns', FunctionTransformer(CustomPipelineUtils.bin_columns)),
        ('scale_num',  ColumnTransformer(MinMaxScaler(), columns=num_cols)),
        ('scale_agr',  ColumnTransformer(MinMaxScaler(), columns=[re.compile('^agr_.*')]))
    ])

    pipeline.fit_transform(df)
    
    return df


def get_xy_cols(df):
    X_col = []
    X_col += num_cols
    X_col += [cc for cc in df.columns if cc.startswith('cat_')]
    X_col += [cc for cc in df.columns if cc.startswith('agr_')]
    y_col = 'Transported'
    
    return X_col, y_col


def get_xy(df):
    X_col, y_col = get_xy_cols(df)
    X = df[X_col].to_numpy()
    if y_col in df.columns:
        y = df[y_col].astype('int').to_numpy()
    else:
        y = None
    return X, y


def run(data_provider: TrainTestDataProvider):
    df = load_and_transform_data()
    data_provider.set_df(df)
    data_provider.save()

class KaggleSpaceshipTitanicDataProvider(TrainTestDataProvider):

    def __init__(self) -> None:
        super().__init__()
        self._train_query = '`train` == 1'
        self._test_query = '`train` == 0'
        self._y_columns = 'Transported'

    def get_x_columns(self) -> List[str]:
        X_col = []
        X_col += num_cols
        X_col += [cc for cc in self._df.columns if cc.startswith('cat_')]
        X_col += [cc for cc in self._df.columns if cc.startswith('agr_')]
        return [ c for c in X_col if c not in ['VIP','agr_Sum_Expenses','agr_Count_Name_Last','agr_Count_Cabin_Num']]

    def get_y_train(self) -> ndarray:
        return self._get_df().query(self._train_query)[self.get_y_columns()].astype('int').to_numpy()