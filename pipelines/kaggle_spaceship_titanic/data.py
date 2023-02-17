import os
import re
import numpy as np
import pandas as pd

from typing import List
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
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

    def del_dummies(df, columns: List[str]):
        for c in columns:
            dummies = pd.get_dummies(df[c], prefix=f'cat_{c}', drop_first=False)
            new_columns += list(dummies.columns)
            df.drop(new_columns, axis=1, inplace=True)
        return df

class CustomPipelineUtils:
    
    def add_split_columns(df):
        df[['PassengerId_Group','PassengerId_Number']] = df['PassengerId'].str.split('_', expand=True)
        df[['Cabin_Deck','Cabin_Num','Cabin_Side']] = df['Cabin'].str.split('/', expand=True)
        df[['Name_First','Name_Last']] = df['Name'].str.split(' ', expand=True)
        return df

    def fillna_insights(df):
        df['VIP'] = df['VIP'].astype('str').replace('nan',np.nan)
        df['CryoSleep'] = df['CryoSleep'].astype('str').replace('nan',np.nan)
        df['_Paid_Services'] = 'True'
        df.loc[df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1) == 0, '_Paid_Services'] = 'False'

        # From axent tool        
        df.loc[df.query("`Cabin_Deck` == 'A'").index, "HomePlanet"] = "Europa"
        df.loc[df.query("`Cabin_Deck` == 'B'").index, "HomePlanet"] = "Europa"
        df.loc[df.query("`Cabin_Deck` == 'C'").index, "HomePlanet"] = "Europa"
        df.loc[df.query("`Cabin_Deck` == 'G'").index, "HomePlanet"] = "Earth"
        df.loc[df.query("`Cabin_Deck` == 'T'").index, "HomePlanet"] = "Europa"
        df.loc[df.query("`HomePlanet` == 'Earth'").index, "VIP"] = "False"
        df.loc[df.query("`Cabin_Deck` == 'T'").index, "CryoSleep"] = "False"
        df.loc[df.query("`_Paid_Services` == 'True'").index, "CryoSleep"] = "False"
        df.loc[df.query("`Destination` == 'PSO J318.5-22' and `Cabin_Deck` == 'D'").index, "HomePlanet"] = "Mars"
        df.loc[df.query("`Destination` == '55 Cancri e' and `VIP` == 'True'").index, "HomePlanet"] = "Europa"
        df.loc[df.query("`HomePlanet` == 'Mars' and `Destination` == '55 Cancri e'").index, "VIP"] = "False"
        df.loc[df.query("`Cabin_Deck` == 'F' and `VIP` == 'True'").index, "HomePlanet"] = "Mars"
        df.loc[df.query("`HomePlanet` == 'Mars' and `CryoSleep` == 'True'").index, "VIP"] = "False"
        df.loc[df.query("`Destination` == '55 Cancri e' and `Cabin_Deck` == 'F'").index, "VIP"] = "False"
        df.loc[df.query("`Cabin_Deck` == 'F' and `_Paid_Services` == 'False'").index, "VIP"] = "False"
        df.loc[df.query("`HomePlanet` == 'Mars' and `Destination` == 'PSO J318.5-22' and `_Paid_Services` == 'False'").index, "CryoSleep"] = "True"
        df.loc[df.query("`HomePlanet` == 'Europa' and `Cabin_Deck` == 'E' and `Cabin_Side` == 'P'").index, "VIP"] = "False"
        df.loc[df.query("`HomePlanet` == 'Europa' and `Cabin_Deck` == 'E' and `Cabin_Side` == 'P' and `_Paid_Services` == 'False'").index, "CryoSleep"] = "True"
        df.loc[df.query("`Destination` == '55 Cancri e' and `Cabin_Deck` == 'C' and `Cabin_Side` == 'P' and `CryoSleep` == 'True'").index, "VIP"] = "False"
        df.loc[df.query("`Destination` == 'TRAPPIST-1e' and `Cabin_Deck` == 'B' and `Cabin_Side` == 'S' and `CryoSleep` == 'True'").index, "VIP"] = "False"
        df.loc[df.query("`Cabin_Deck` == 'B' and `Cabin_Side` == 'P' and `CryoSleep` == 'False' and `_Paid_Services` == 'False'").index, "Destination"] = "TRAPPIST-1e"
        df.loc[df.query("`Cabin_Deck` == 'C' and `Cabin_Side` == 'P' and `CryoSleep` == 'False' and `_Paid_Services` == 'False'").index, "Destination"] = "TRAPPIST-1e"
        df.loc[df.query("`Destination` == '55 Cancri e' and `Cabin_Deck` == 'A' and `Cabin_Side` == 'S' and `_Paid_Services` == 'False'").index, "CryoSleep"] = "True"
        df.loc[df.query("`Destination` == '55 Cancri e' and `Cabin_Deck` == 'B' and `Cabin_Side` == 'P' and `_Paid_Services` == 'False'").index, "CryoSleep"] = "True"
        df.loc[df.query("`Destination` == '55 Cancri e' and `Cabin_Deck` == 'C' and `Cabin_Side` == 'P' and `_Paid_Services` == 'False'").index, "CryoSleep"] = "True"

        for c in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
            df.loc[df['_Paid_Services']=='False', c] = 0.0

        # HomePlanet and Cabin_Side can be deducted from PassengerId_Group
        for val in df[df['HomePlanet'].isna()]['PassengerId_Group']:
            fill_val_series = df[(df['PassengerId_Group']==val) & (df['HomePlanet'].notna())]['HomePlanet']
            fill_val = fill_val_series.iloc[0] if fill_val_series.shape[0]>0 else np.nan
            df.loc[df['PassengerId_Group']==val,'HomePlanet'] = fill_val

        for val in df[df['Cabin_Side'].isna()]['PassengerId_Group']:
            fill_val_series = df[(df['PassengerId_Group']==val) & (df['Cabin_Side'].notna())]['Cabin_Side']
            fill_val = fill_val_series.iloc[0] if fill_val_series.shape[0]>0 else np.nan
            df.loc[df['PassengerId_Group']==val,'Cabin_Side'] = fill_val

        for val in df[df['HomePlanet'].isna()]['Name_Last']:
            fill_val_series = df[(df['Name_Last']==val) & (df['HomePlanet'].notna())]['HomePlanet']
            fill_val = fill_val_series.iloc[0] if fill_val_series.shape[0]>0 else np.nan
            df.loc[df['Name_Last']==val,'Name_Last'] = fill_val

        return df
                
    def fillna_after_impute(df):
        cd_cols = [ c for c in df.columns if c.startswith('cat_Cabin_Deck_') ]
        df['_Cabin_Deck'] = df['Cabin_Deck']
        df.loc[df['Cabin'].isna(), '_Cabin_Deck'] = df[cd_cols].idxmax(axis=1).str.replace('cat_Cabin_Deck_','')

        cd_cols = [ c for c in df.columns if c.startswith('cat_Cabin_Side_') ]
        df['_Cabin_Side'] = df['Cabin_Side']
        df.loc[df['Cabin'].isna(), '_Cabin_Side'] = df[cd_cols].idxmax(axis=1).str.replace('cat_Cabin_Side_','')

        df['_Cabin_Deck_Side'] = df['_Cabin_Deck'].astype('str') + "/" + df['_Cabin_Side'].astype('str')
        for cds in df['_Cabin_Deck_Side'].unique():
            df_slice = df[df['_Cabin_Deck_Side']==cds]
            a,b = np.polyfit(df_slice['PassengerId_Group'], df_slice['Cabin_Num'], deg=1)
            df.loc[(df['_Cabin_Deck_Side']==cds) & (df['Cabin'].isna()), 'Cabin_Num'] = a * df['PassengerId_Group'] + b

        return df

    def log_scale(df):
        cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        for c in cols:
            df[c] = np.log(df[c]+1.0)
        return df

    def _bin_column(df, column, bins, right=False):
        df[f'bin_{column}'] = pd.cut(df[column],bins=bins, right=right)
        dummies = pd.get_dummies(df[f'bin_{column}'], prefix=f'cat_bin_{column}')
        df[list(dummies.columns)[:-1]] = dummies[list(dummies.columns)[:-1]]
        return df

    def bin_columns(df):
        df = CustomPipelineUtils._bin_column(df, 'Age', [0,12,20,45,800])
        df = CustomPipelineUtils._bin_column(df, 'Cabin_Num', list(range(0,2400,300)))
        # df = CustomPipelineUtils._bin_column(df, 'PassengerId_Group', list(range(0,12000,1000)))
        return df
    
    def add_summary_columns(df):
        # df['agr_Count_Cabin_Num'] = df.groupby(['Cabin_Num']).transform('count')['PassengerId'].fillna(0.0)
        # df['agr_Sum_Expenses'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
        # df['agr_Count_PassengerId_Group'] = df.groupby(['PassengerId_Group']).transform('count')['PassengerId'].fillna(0.0)
        # df['agr_Count_Name_Last'] = df.groupby(['Name_Last']).transform('count')['PassengerId'].fillna(0.0)
        return df

cat_cols = ['HomePlanet','CryoSleep','Destination','VIP','Cabin_Deck','Cabin_Side']
num_cols = ['Age','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','Cabin_Num','PassengerId_Group','PassengerId_Number']

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
        ('add_dummies', FunctionTransformer(PipelineUtils.add_dummies, kw_args={'columns':cat_cols, 'drop_first':False})),
        ('impute', ColumnTransformer(KNNImputer(weights='distance'), columns=num_cols+[re.compile('^cat_.*')])),
        ('fillna_after_impute', FunctionTransformer(CustomPipelineUtils.fillna_after_impute)),
        ('add_summary_columns', FunctionTransformer(CustomPipelineUtils.add_summary_columns)),
        ('bin_columns', FunctionTransformer(CustomPipelineUtils.bin_columns)),
        ('scale_num',  ColumnTransformer(MinMaxScaler(), columns=num_cols)),
        # ('scale_agr',  ColumnTransformer(MinMaxScaler(), columns=[re.compile('^agr_.*')]))
    ])

    pipeline.fit_transform(df)
    
    return df[df['train']==1], df[df['train']==0].copy()


def get_xy_cols(df):
    X_col = []
    X_col += num_cols
    X_col += [cc for cc in df.columns if cc.startswith('cat_')]
    X_col += [cc for cc in df.columns if cc.startswith('agr_')]
    X_col = [ c for c in X_col if c not in ['cat_Cabin_Deck_T', 'cat_Cabin_Side_S', 'cat_VIP_True', 'cat_Destination_TRAPPIST-1e', 'cat_CryoSleep_True', 'cat_HomePlanet_Mars', 'cat_bin_Cabin_Num_[0, 300)'] ]
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