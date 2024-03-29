from datetime import datetime
from airflow import DAG, Dataset, XComArg
from airflow.operators.python import PythonOperator

from axent.common.data import TrainTestDataProvider
from axent.common.data import FeaturesDataProvider
from axent.common.runtime import FeatureSelection
from axent.common.task import ModelTrainBestTask
from axent.common.task import ModelHyperparameterTuningTask
from axent.kaggle_spaceship_titanic.data import run as prepare_data
from axent.common.model import CatBoostClassifierModel

##### Inititalize globals #####

n_trials = 10


##### Inititalize datasets and data providers #####

train_test_dataset = Dataset('s3://kaggle-spaceship-titanic/data/train-test.csv')
features_dataset = Dataset('s3://kaggle-spaceship-titanic/features')

dp = TrainTestDataProvider()
dp.set_uri(train_test_dataset.uri)
dp.set_y_columns('Transported')
dp.set_train_column('train')

fp = FeaturesDataProvider()
fp.set_base_uri(features_dataset.uri)


##### Callables #####

def feature_selection_to_models_callable(ti):
    output = []
    feature_selection_outputs = ti.xcom_pull(key="return_value", task_ids="feature-selection")
    for feature_selection_output in feature_selection_outputs:
        for model_name in [CatBoostClassifierModel.__name__]:
            model_hyperparameter_tuning_input = {}
            model_hyperparameter_tuning_input['model_name'] = model_name
            model_hyperparameter_tuning_input['features_class'] = feature_selection_output['features_class']
            model_hyperparameter_tuning_input['n_trials'] = n_trials
            output.append(model_hyperparameter_tuning_input)
    return output


##### DAGs #####

with DAG(
    "kaggle-spaceship-titanic-data",
    description="Kaggle 'Spaceship Titanic'",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag_data:

    prepare_data_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
        outlets=[train_test_dataset],
        op_kwargs={
            'data_provider' : dp
        }
    )


with DAG(
    "kaggle-spaceship-titanic-model",
    description="Kaggle 'Spaceship Titanic'",
    start_date=datetime(2023, 1, 1),
    schedule=[train_test_dataset],
    schedule_interval=None,
    catchup=False
) as dag_model:

    feature_selection_tasks = PythonOperator.partial(
        task_id='feature-selection',
        python_callable=FeatureSelection(data_provider=dp, features_provider=fp).run
    ).expand(
        op_kwargs = [
            { 'mode':'chi2_cap_corr', 'label':'chi2-corr60', 'max_correlation':0.6 },
            { 'mode':'chi2_cap_corr', 'label':'chi2-corr40-no-vip', 'max_correlation':0.4, 'excluded_features':['VIP'] },
            { 'mode':'chi2_cap_corr', 'label':'chi2-corr60-no-vip', 'max_correlation':0.6, 'excluded_features':['VIP'] },
            { 'mode':'chi2_cap_corr', 'label':'chi2-corr80-no-vip', 'max_correlation':0.8, 'excluded_features':['VIP'] },
            { 'mode':'chi2_k_best', 'label':'chi2-best20', 'k':20 },
            { 'mode':'chi2_k_best', 'label':'chi2-best22', 'k':22 }
        ]
    )

    feature_selection_to_models_task = PythonOperator(
        task_id='feature-selection-to-models',
        python_callable=feature_selection_to_models_callable
    )

    model_hyperparameter_tuning_tasks = PythonOperator.partial(
        task_id='model-hyperparameter-tuning',
        python_callable=ModelHyperparameterTuningTask(data_provider=dp, features_provider=fp).execute
    ).expand(
        op_kwargs = XComArg(feature_selection_to_models_task, key='return_value')
    )

    model_train_best_task = PythonOperator(
        task_id='model-train-best',
        python_callable=ModelTrainBestTask(data_provider=dp, features_provider=fp).execute
    )

    feature_selection_tasks >> feature_selection_to_models_task >> model_hyperparameter_tuning_tasks >> model_train_best_task
