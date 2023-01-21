#!/bin/bash

export MLFLOW_TRACKING_URI=http://${MLFLOW_HOST}:${MLFLOW_PORT}
export MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://${DB_MLFLOW_USER}:${DB_MLFLOW_PASS}@${DB_HOST}/${DB_MLFLOW_NAME}
export MLFLOW_S3_ENDPOINT_URL=http://${S3_HOST}:${S3_PORT}
export MLFLOW_ARTIFACTS_DESTINATION=s3://${S3_BUCKET}/
export AWS_ACCESS_KEY_ID=$S3_AUTH_KEY_ID
export AWS_SECRET_ACCESS_KEY=$S3_AUTH_SECRET_KEY
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://${DB_AIRFLOW_USER}:${DB_AIRFLOW_PASS}@${DB_HOST}/${DB_AIRFLOW_NAME}"
export AIRFLOW__DATABASE__SQL_ALCHEMY_SCHEMA="public"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_FOLDER=/home/airflow/pipelines
export PATH=/home/airflow/.local/bin:$PATH

until airflow db check; do echo "[INFO ]: Waiting for database"; sleep 5; done;
airflow db init
airflow users create \
    -u ${AF_USER} \
    -f Air \
    -l Flow \
    -p ${AF_PASS} \
    -r Admin \
    -e airlow@axent.pl
airflow scheduler &
airflow webserver --port 8080 &
wait
