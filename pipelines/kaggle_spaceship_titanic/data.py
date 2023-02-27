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
from .common.transformer.fillna_association_transformer import FillnaAssociationTransformer
from .common_data import TrainTestDataProvider

cat_cols = ['HomePlanet','Destination','Cabin_Deck','Cabin_Side']
num_cols = ['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Cabin_Num','PassengerId_Group','PassengerId_Number','CryoSleep','VIP']
exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

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


class CustomPipelineUtils:
    
    def add_columns(df):
        df[['PassengerId_Group','PassengerId_Number']] = df['PassengerId'].str.split('_', expand=True)
        df[['Cabin_Deck','Cabin_Num','Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
        df[['Name_First','Name_Last']] = df['Name'].str.split(' ', expand=True)

        
        df['_has_expenses'] = (df[exp_cols].max(axis=1) > 0)
        df['_has_expenses'] = df['_has_expenses'].astype(float)
        for c in exp_cols:
            df.loc[(df[c].isna()) & (df['_has_expenses'] == 0), '_has_expenses'] = np.nan

        return df

    def fillna_expenses(df):
        # fill expenses after fillna_associations
        for c in exp_cols:
            df.loc[(df[c].isna()) & (df['_has_expenses'] == 0), c] = 0.0
        return df

    def fillna_imputer(df):
        # Mark which Cabin_Num values were filled with associations
        df['_Cabin_Num_Na'] = 0
        df.loc[df['Cabin_Num'].isna(),'_Cabin_Num_Na'] = 1

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
            df.loc[(df['Cabin_Deck_Side']==cds) & (df['_Cabin_Num_Na']==1), 'Cabin_Num'] = a * df['PassengerId_Group'] + b
        return df

    def log_scale(df):
        cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        cols += [ c for c in df.columns if 'Expenses' in c ]
        for c in cols:
            df[c] = np.log(df[c]+1.0)
        return df

    def _bin_column(df, column, bins, right=False):
        df[f'bin_{column}'] = pd.cut(df[column],bins=bins, right=right)
        dummies = pd.get_dummies(df[f'bin_{column}'], prefix=f'cat_bin_{column}')
        df[list(dummies.columns)[1:]] = dummies[list(dummies.columns)[1:]]
        return df

    def bin_columns(df):
        df = CustomPipelineUtils._bin_column(df, 'Age', [0,13,18,26,29,49,200])
        df = CustomPipelineUtils._bin_column(df, 'Cabin_Num', list(range(0,2400,300)))
        df = CustomPipelineUtils._bin_column(df, 'PassengerId_Group', list(range(0,11000,1000)))
        return df
    
    def add_summary_columns(df):
        df['agr_Sum_Expenses'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1).fillna(0.0)
        df["agr_Sum_Expenses_Regular"] = df["FoodCourt"] + df["ShoppingMall"] 
        df["agr_Sum_Expenses_Luxury"] = df["RoomService"] + df["Spa"] + df["VRDeck"]

        df['_Cabin'] = df['Cabin_Deck'].astype('str') + "/" + df['Cabin_Side'].astype('str') + "/" + df['Cabin_Num'].astype('str')
        df['agr_Count_PassengerId_by_Cabin'] = df.groupby(['_Cabin'])['PassengerId'].transform('count').fillna(0.0)
        df['agr_Count_PassengerId_by_Group'] = df.groupby(['PassengerId_Group'])['PassengerId'].transform('count').fillna(0.0)

        df['agr_Avg_Expenses_by_Deck'] = df.groupby(['Cabin_Deck'])['agr_Sum_Expenses'].transform('mean').fillna(0.0)

        df['agr_Sum_Expenses_by_Cabin'] = df.groupby(['_Cabin'])['agr_Sum_Expenses'].transform('sum').fillna(0.0)
        df['agr_Sum_Expenses_by_Group'] = df.groupby(['PassengerId_Group'])['agr_Sum_Expenses'].transform('sum').fillna(0.0)

        df['agr_Age_Cabin_Num'] = df['Age'] * df['Cabin_Num']
        df['agr_Age_Expenses'] = df['Age'] * df['agr_Sum_Expenses']

        return df


def load_and_transform_data():
    df_train = pd.read_csv(os.path.dirname(__file__)+'/train.csv')
    df_train['train'] = 1

    df_test = pd.read_csv(os.path.dirname(__file__)+'/test.csv')
    df_test['train'] = 0

    df = pd.concat([df_test, df_train], ignore_index=True)

    pipeline = Pipeline(steps=[
        ('add_columns', FunctionTransformer(CustomPipelineUtils.add_columns)),
        ('astype_num', ColumnTransformer(TypeTransformer('float64'), columns=num_cols+['Transported'])),
        ('fillna_associations', ColumnTransformer(FillnaAssociationTransformer(), columns=['HomePlanet', 'Destination', 'CryoSleep', 'VIP', 'Age', 'Cabin_Deck', 'Cabin_Num', 'Cabin_Side', '_has_expenses', 'PassengerId_Group'])),
        ('fillna_expenses', FunctionTransformer(CustomPipelineUtils.fillna_expenses)),
        ('fillna_imputer', FunctionTransformer(CustomPipelineUtils.fillna_imputer)),
        ('fillna_cabin_num', FunctionTransformer(CustomPipelineUtils.fillna_fix_cabin_num)),
        ('astype_cat', ColumnTransformer(TypeTransformer('category'), columns=cat_cols)),
        ('add_dummies', FunctionTransformer(PipelineUtils.add_dummies, kw_args={'columns':cat_cols})),
        ('add_summary_columns', FunctionTransformer(CustomPipelineUtils.add_summary_columns)),
        ('log_scale', FunctionTransformer(CustomPipelineUtils.log_scale)),
        ('bin_columns', FunctionTransformer(CustomPipelineUtils.bin_columns)),
        ('scale_num',  ColumnTransformer(MinMaxScaler(), columns=num_cols)),
        ('scale_agr',  ColumnTransformer(MinMaxScaler(), columns=[re.compile('^agr_.*')]))
    ])

    pipeline.fit_transform(df)
    
    for c in df.columns:
        if c.startswith('_'):
            df.drop(c, axis=1, inplace=True)

    return df


def run(data_provider: TrainTestDataProvider):
    df = load_and_transform_data()
    data_provider.set_df(df)
    data_provider.save()

