import os
import re
import numpy as np
import pandas as pd

from typing import List
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline

from .column_transformer import ColumnTransformer
from .type_transformer import TypeTransformer

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
    
    def add_split_columns(df):
        df[['PassengerId_Group','PassengerId_Number']] = df['PassengerId'].str.split('_', expand=True)
        df[['Cabin_Deck','Cabin_Num','Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
        df[['Name_First','Name_Last']] = df['Name'].str.split(' ', expand=True)
        return df

    def fillna_insights(df):
        # Sleepers and passengers under 13 do not spend money
        for c in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
            df.loc[(df['CryoSleep']==True) | (df['Age']<13), c] = 0.0
        # No VIPs from Earth
        df.loc[df['HomePlanet']=='Earth', 'VIP'] = False
        return df
                
    def log_scale(df):
        cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for c in cols:
            df[c] = np.log(df[c]+1.0)
        return df

    def bin_columns(df):
        age_bins = [-1,12,18,25,30,50,800]
        df['bin_Age'] = pd.cut(df['Age'],bins=age_bins)
        dummies = pd.get_dummies(df['bin_Age'], prefix=f'cat_bin_Age')
        df[list(dummies.columns)[:-1]] = dummies[list(dummies.columns)[:-1]]
        return df
    
    def add_summary_columns(df):
        df['agr_Count_Cabin_Num'] = df.groupby(['Cabin_Num']).transform('count')['PassengerId'].fillna(0.0)
        df['agr_Count_PassengerId_Group_Name_Last'] = df.groupby(['PassengerId_Group','Name_Last']).transform('count')['PassengerId'].fillna(0.0)
        df['agr_Alone'] = 0
        df.loc[df['agr_Count_PassengerId_Group_Name_Last']<2,'agr_Alone'] = 1
        df.drop(columns=['agr_Count_PassengerId_Group_Name_Last'], inplace=True)
        return df

cat_cols = ['HomePlanet','CryoSleep','Destination','VIP','Cabin_Deck','Cabin_Side','PassengerId_Number']
num_cols = ['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Cabin_Num','PassengerId_Group']

def load_and_transform_data():
    df_train = pd.read_csv(os.path.dirname(__file__)+'/train.csv')
    df_train['train'] = 1

    df_test = pd.read_csv(os.path.dirname(__file__)+'/test.csv')
    df_test['train'] = 0

    df = pd.concat([df_test, df_train], ignore_index=True)

    pipeline = Pipeline(steps=[
        ('split', FunctionTransformer(CustomPipelineUtils.add_split_columns)),
        ('fillna_insights', FunctionTransformer(CustomPipelineUtils.fillna_insights)),
        ('astype_num', ColumnTransformer(TypeTransformer('float64'), columns=num_cols)),
        ('log_scale', FunctionTransformer(CustomPipelineUtils.log_scale)),
        ('astype_cat', ColumnTransformer(TypeTransformer('category'), columns=cat_cols)),
        ('add_dummies', FunctionTransformer(PipelineUtils.add_dummies, kw_args={'columns':cat_cols})),
        ('impute', ColumnTransformer(KNNImputer(), columns=num_cols+[re.compile('^cat_.*')])),
        ('bin_columns', FunctionTransformer(CustomPipelineUtils.bin_columns)),
        ('add_summary_columns', FunctionTransformer(CustomPipelineUtils.add_summary_columns)),
        ('scale_num',  ColumnTransformer(MinMaxScaler(), columns=num_cols)),
        ('scale_agr',  ColumnTransformer(MinMaxScaler(), columns=[re.compile('^agr_.*')]))
    ])

    pipeline.fit_transform(df)
    
    return df[df['train']==1], df[df['train']==0].copy()


def get_xy_cols(df):
    X_col = []
    X_col += num_cols
    X_col += [cc for cc in df.columns if cc.startswith('cat_')]
    X_col += [cc for cc in df.columns if cc.startswith('agr_')]
    X_col = [ c for c in X_col if c not in ['Age','Cabin_Num'] ]
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

def run():
    return [os.path.dirname(__file__)+'/train.csv', os.path.dirname(__file__)+'/test.csv']