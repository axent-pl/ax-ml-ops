from datetime import datetime
from airflow import DAG, Dataset, XComArg
from airflow.operators.python import PythonOperator

from kaggle_spaceship_titanic.data import run as prepare_data
from kaggle_spaceship_titanic.common_model import ModelClass
from kaggle_spaceship_titanic.common_data import TrainTestDataProvider
from kaggle_spaceship_titanic.common.task.model_hyperparameter_tuning import ModelHyperParameterTuning
from kaggle_spaceship_titanic.common.task.feature_selection import FeatureSelection


##### Inititalize globals #####

version = 'v13'
n_trials = 100


##### Inititalize datasets and data providers #####

train_tesst_dataset = Dataset('s3://kaggle-spaceship-titanic/data/train-test.csv')
dp = TrainTestDataProvider()
dp.set_uri(train_tesst_dataset.uri)
dp.set_y_columns('Transported')
dp.set_train_column('train')


##### Inititalize task processors #####

feature_selection = FeatureSelection(data_provider=dp)
model_hyperparameter_tuning = ModelHyperParameterTuning(data_provider=dp)


##### Callables #####

def feature_selection_to_models_callable(ti):
    output = []
    feature_selection_outputs = ti.xcom_pull(key="return_value", task_ids="feature-selection")
    for feature_selection_output in feature_selection_outputs:
        for model_class in [ModelClass.CBC]:
            model_hyperparameter_tuning_input = feature_selection_output.copy()
            model_hyperparameter_tuning_input['model_class'] = model_class
            model_hyperparameter_tuning_input['label'] = f"{model_class.value.lower()}-{model_hyperparameter_tuning_input['label']}-{version}"
            model_hyperparameter_tuning_input['n_trials'] = n_trials
            output.append(model_hyperparameter_tuning_input)
    return output

def model_ensemble_callable(ti):
    return 'ensembled'


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

    feature_selection_tasks = PythonOperator.partial(
        task_id='feature-selection',
        python_callable=feature_selection.run
    ).expand(
        op_kwargs = [
            { 'mode':'chi2_cap_corr', 'label':'chi2_cap_corr', 'max_correlation':0.6 },
            { 'mode':'chi2_k_best', 'label':'chi2_k_best', 'k':20 }
            # { 'mode':'fclassif_max_corr', 'label':'f-classif', 'max_correlation':0.6, 'excluded_features':['VIP'] }
            # { 'mode':'mutualinfoclassif_max_corr', 'label':'mi-classif' }
        ]
    )

    feature_selection_to_models_task = PythonOperator(
        task_id='feature-selection-to-models',
        python_callable=feature_selection_to_models_callable
    )

    model_hyperparameter_tuning_tasks = PythonOperator.partial(
        task_id='model-hyperparameter-tuning',
        python_callable=model_hyperparameter_tuning.run
    ).expand(
        op_kwargs = XComArg(feature_selection_to_models_task, key='return_value')
    )

    model_ensemble_task = PythonOperator(
        task_id='model-ensemble',
        python_callable=model_ensemble_callable,
        inlets=[train_tesst_dataset]
    )

    feature_selection_tasks >> feature_selection_to_models_task >> model_hyperparameter_tuning_tasks >> model_ensemble_task
