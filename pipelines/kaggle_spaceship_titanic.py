from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from kaggle_spaceship_titanic.data import run as prepare_data
from kaggle_spaceship_titanic.common_model import ModelClass
from kaggle_spaceship_titanic.common import model_hyperparameter_tuning_task
from kaggle_spaceship_titanic.data import KaggleSpaceshipTitanicDataProvider

with DAG(
    "kaggle-spaceship-titanic",
    description="Kaggle 'Spaceship Titanic'",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    version = 'v5'
    n_trials = 10

    prepare_data_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    model_train_tasks = []
    dp = KaggleSpaceshipTitanicDataProvider()
    for mc in ModelClass:
        model_train_tasks.append(PythonOperator(
        task_id=f'train-{mc.value}',
        python_callable=model_hyperparameter_tuning_task,
        op_kwargs={
            'task_name':f'train-{mc.value}-{version}',
            'model_class' : mc,
            'data_provider' : dp,
            'n_trials': n_trials
        }
    ))

    prepare_data_task >> model_train_tasks
