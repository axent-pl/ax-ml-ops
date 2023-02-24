from ...common_data import TrainTestDataProvider

import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif

class FeatureSelection:

    def __init__(self, data_provider: TrainTestDataProvider = None) -> None:
        self._dp:TrainTestDataProvider = data_provider
        pass

    def _get_all_features(self):
        dp_train_df = self._dp.get_train_dataframe()
        all_features = []
        for c in dp_train_df.columns:
            if self._dp.can_be_x_column(c):
                all_features.append(c)
        return all_features

    def _run_selector_max_corr(self, selector:str, max_correlation:float = 0.5):
        all_features = self._get_all_features()
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

    def run_chi2_max_corr(self, max_correlation:float = 0.5, *args, **kwargs):
        return self._run_selector_max_corr('chi2', max_correlation=max_correlation)

    def run_fclassif_max_corr(self, max_correlation:float = 0.5, *args, **kwargs):
        return self._run_selector_max_corr('f_classif', max_correlation=max_correlation)

    def run_mutualinfoclassif_max_corr(self, max_correlation:float = 0.5, *args, **kwargs):
        return self._run_selector_max_corr('mutual_info_classif', max_correlation=max_correlation)

    def run(self, mode: str, *args, **kwargs):
        if hasattr(self, f'run_{mode}'):
            return {
                'feature_selection_label': kwargs['label'] if 'label' in kwargs else mode,
                'feature_selection_mode': mode,
                'x_columns': getattr(self, f'run_{mode}')(*args, **kwargs)
            }
        raise Exception(f'Mode {mode} not supported')