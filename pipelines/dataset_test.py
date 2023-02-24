from typing import List
from airflow import DAG, Dataset, XComArg
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models.taskinstance import TaskInstance
from datetime import datetime

from kaggle_spaceship_titanic.data import TrainTestDataProvider
from kaggle_spaceship_titanic.data import KaggleSpaceshipTitanicDataProvider

# Define datasets
train_tesst_dataset = Dataset('s3://dynamic-tasks/data/train-test.csv')
dp = KaggleSpaceshipTitanicDataProvider()
dp.set_uri(train_tesst_dataset.uri)
version = 'v1'


def model_feature_selection(version: str = None, data_provider: TrainTestDataProvider = None,  *args, **kwargs):
    outputs = [
        { 'x_columns':['X1'] },
        { 'x_columns':['X1','X2'] },
        { 'x_columns':['X2','X3'] }
    ]
    return outputs

def model_hyperparameter_tuning(version: str = None, model_class: str = 'm', data_provider: TrainTestDataProvider = None, n_trials: int = 100, n_splits: int = 5, scoring: str = 'accuracy', direction = 'maximize', x_columns = None, *args, **kwargs):
    return {'x_columns': x_columns, 'model_class': model_class, 'data_provider': data_provider.get_y_columns() }

def model_ensemble():
    return 'ensembled'

with DAG(
    dag_id='dynamic-tasks-v2',
    start_date=datetime(2021, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag1:

    ##### CALLABLES #####

    def model_feature_selection_collable(*args, **kwargs):
        return model_feature_selection(version=version, data_provider=dp)

    def model_hyperparameter_tuning_fanout_callable(ti):
        outputs = []
        feature_selection_outputs = ti.xcom_pull(key="return_value", task_ids="data-feature-selection")
        for feature_selection_output in feature_selection_outputs:
            for m in ['cbc','svc','rfc']:
                output = feature_selection_output.copy()
                output['model_class'] = m
                outputs.append(output)
        return outputs

    def model_hyperparameter_tuning_callable(model_class: str, x_columns = None, *args, **kwargs):
        return model_hyperparameter_tuning(version=version, model_class=model_class, data_provider=dp, x_columns=x_columns)

    ##### TASKS #####

    model_feature_selection_task = PythonOperator(
        task_id='data-feature-selection',
        python_callable=model_feature_selection_collable,
    )

    model_hyperparameter_tuning_fanout_task = PythonOperator(
        task_id=f'model-hyperparameter-tuning-fanout',
        python_callable=model_hyperparameter_tuning_fanout_callable
    )

    model_hyperparameter_tuning_tasks = PythonOperator.partial(
        task_id=f'model-hyperparameter-tuning',
        python_callable=model_hyperparameter_tuning_callable
    ).expand(
        op_kwargs = XComArg(model_hyperparameter_tuning_fanout_task, key='return_value')
    )

    model_ensemble_task = PythonOperator(
        task_id='model-ensemble',
        python_callable=model_ensemble
    )

    model_feature_selection_task >> model_hyperparameter_tuning_fanout_task >> model_hyperparameter_tuning_tasks >> model_ensemble_task