from typing import List
from airflow import DAG, Dataset, XComArg
from airflow.decorators import task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models.taskinstance import TaskInstance
from datetime import datetime

with DAG(
    dag_id='xcom-test',
    start_date=datetime(2021, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag1:

    def feature_selection_callable(mode:str):
        return {
            'mode': mode,
            'x_columns': [1,2,3]
        }

    def feature_selection_fanin_callable(ti):
        feature_selection_outputs = ti.xcom_pull(key="return_value", task_ids="feature-selection")
        return feature_selection_outputs

    feature_selection_tasks = PythonOperator.partial(
        task_id = 'feature-selection',
        python_callable = feature_selection_callable
    ).expand(
        op_kwargs = [
            {'mode':'chi2corr'},
            {'mode':'chi2corr'},
            {'mode':'chi2corr'}
        ]
    )

    feature_selection_fanin_task = PythonOperator(
        task_id = 'feature-selection-fan-in',
        python_callable = feature_selection_fanin_callable
    )


    feature_selection_tasks >> feature_selection_fanin_task