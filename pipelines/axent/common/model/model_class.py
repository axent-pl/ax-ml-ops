from enum import Enum


class ModelClass(str, Enum):
    CBC = 'CatBoostClassifier'
    RFC = 'RandomForestClassifier'
    SVC = 'SVC'
    LRC = 'LogisticRegression'
