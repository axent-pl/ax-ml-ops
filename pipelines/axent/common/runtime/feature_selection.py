from typing import List
from ..data import TrainTestDataProvider
from ..data import FeaturesDataProvider

import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

class FeatureSelection:

    def __init__(self, data_provider: TrainTestDataProvider = None, features_provider: FeaturesDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        self._fp:FeaturesDataProvider = features_provider

    def _get_all_features(self, excluded_features:List[str] = []):
        dp_train_df = self._dp.get_train_dataframe()
        all_features = []
        for c in dp_train_df.columns:
            if c not in excluded_features and self._dp.can_be_x_column(c) and not dp_train_df[c].isnull().values.any():
                all_features.append(c)
        return all_features

    def _run_selector_cap_corr(self, selector:str, max_correlation:float = 0.5, excluded_features:List[str] = []):
        all_features = self._get_all_features(excluded_features=excluded_features)
        X = self._dp.get_train_dataframe()[all_features].to_numpy()
        y = self._dp.get_y_train()

        if selector == 'chi2':
            fs_scores,_ = chi2(X, y)
        elif selector == 'f_classif':
            fs_scores,_ = f_classif(X, y)
        elif selector == 'mutual_info_classif':
            fs_scores = mutual_info_classif(X, y)
        else:
            raise Exception(f'Selector {selector} not supported')

        fs_scores_df = pd.DataFrame({'feature': all_features,'score': fs_scores}).set_index('feature').sort_values(by='score',ascending=False)
        cm_scores = self._dp.get_train_dataframe()[all_features].corr().abs()

        removed_featues = set()
        checked_features = set()
        for f in fs_scores_df.index:
            if f not in removed_featues:
                for ff in cm_scores.query(f'`{f}` > {max_correlation} and `{f}` != 1').index:
                    if ff not in checked_features:
                        removed_featues.add(ff)
            checked_features.add(f)

        return [ f for f in all_features if f not in removed_featues ]

    def run_chi2_cap_corr(self, max_correlation:float = 0.5, excluded_features:List[str] = [], *args, **kwargs):
        return self._run_selector_cap_corr('chi2', max_correlation=max_correlation, excluded_features=excluded_features)

    def run_chi2_k_best(self, k:int, excluded_features:List[str] = [], *args, **kwargs):
        all_features = self._get_all_features(excluded_features=excluded_features)
        X = self._dp.get_train_dataframe()[all_features].to_numpy()
        y = self._dp.get_y_train()
        feature_scores = SelectKBest(chi2, k=k).fit(X, y)
        feature_indx = feature_scores.get_support(indices=True)
        features = self._dp.get_train_dataframe()[all_features].columns[feature_indx]
        return list(features)

    def run_fclassif_cap_corr(self, max_correlation:float = 0.5, excluded_features:List[str] = [], *args, **kwargs):
        return self._run_selector_cap_corr('f_classif', max_correlation=max_correlation, excluded_features=excluded_features)

    def run_mutualinfoclassif_cap_corr(self, max_correlation:float = 0.5, excluded_features:List[str] = [], *args, **kwargs):
        return self._run_selector_cap_corr('mutual_info_classif', max_correlation=max_correlation, excluded_features=excluded_features)

    def run(self, mode: str, *args, **kwargs):
        if hasattr(self, f'run_{mode}'):
            features = getattr(self, f'run_{mode}')(*args, **kwargs)
            features_class = kwargs['label'] if 'label' in kwargs else mode
            self._fp.set_features(features_class=features_class, features=features)
            return {
                'features_class': features_class
            }
        raise Exception(f'Mode {mode} not supported')