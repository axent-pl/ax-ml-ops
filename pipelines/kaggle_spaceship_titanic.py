from datetime import datetime
from airflow import DAG, Dataset
from airflow.operators.python import PythonOperator

from kaggle_spaceship_titanic.data import run as prepare_data
from kaggle_spaceship_titanic.common_model import ModelClass
from kaggle_spaceship_titanic.common_task import model_hyperparameter_tuning_task
from kaggle_spaceship_titanic.data import KaggleSpaceshipTitanicDataProvider

def model_ensemble(ti):
    return 'ensembled'

version = 'v9'
n_trials = 100
train_tesst_dataset = Dataset('s3://kaggle-spaceship-titanic/data/train-test.csv')
dp = KaggleSpaceshipTitanicDataProvider()
dp.set_uri(train_tesst_dataset.uri)

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
        outlets=[train_tesst_dataset],
        op_kwargs={
            'data_provider' : dp
        }
    )

with DAG(
    "kaggle-spaceship-titanic-model",
    description="Kaggle 'Spaceship Titanic'",
    start_date=datetime(2023, 1, 1),
    schedule=[train_tesst_dataset],
    schedule_interval=None,
    catchup=False
) as dag_model:

    model_train_tasks = []
    for mc in ModelClass:
        model_train_tasks.append(PythonOperator(
        task_id=f'train-{mc.value}',
        python_callable=model_hyperparameter_tuning_task,
        inlets=[train_tesst_dataset],
        op_kwargs={
            'task_name':f'train-{mc.value}-{version}',
            'model_class' : mc,
            'data_provider' : dp,
            'n_trials': n_trials
        }
    ))

    model_ensemble_task = PythonOperator(
        task_id='model-ensemble',
        python_callable=model_ensemble,
        inlets=[train_tesst_dataset]
    )

    model_train_tasks >> model_ensemble_task
