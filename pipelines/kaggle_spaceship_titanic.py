from datetime import datetime, timedelta
from textwrap import dedent
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from kaggle_spaceship_titanic.task_train_cbc import run as train_cbc
from kaggle_spaceship_titanic.task_train_rfc import run as train_rfc
from kaggle_spaceship_titanic.task_train_svc import run as train_svc
from kaggle_spaceship_titanic.data import run as prepare_data

with DAG(
    "kaggle-spaceship-titanic",
    default_args={
        "depends_on_past": False,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="Kaggle 'Spaceship Titanic'",
    schedule_interval=None,
    start_date=datetime(2021, 1, 1),
    catchup=False
) as dag:

    prepare_data_task = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
    )

    train_cbc_task = PythonOperator(
        task_id="train_cbc",
        python_callable=train_cbc,
    )

    train_rfc_task = PythonOperator(
        task_id="train_rfc",
        python_callable=train_rfc,
    )

    train_svc_task = PythonOperator(
        task_id="train_svc",
        python_callable=train_svc,
    )

    prepare_data_task >> [train_cbc_task, train_rfc_task, train_svc_task]
