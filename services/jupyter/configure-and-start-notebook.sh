#!/bin/bash
export MLFLOW_TRACKING_URI=http://mlflow:5000
export MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlops:${DB_MLOPS_PASS}@db/mlops
export MLFLOW_S3_ENDPOINT_URL=http://datastore-s3:9000
export MLFLOW_ARTIFACTS_DESTINATION=s3://${S3_BUCKET}/
export AWS_ACCESS_KEY_ID=$S3_AUTH_KEY_ID
export AWS_SECRET_ACCESS_KEY=$S3_AUTH_SECRET_KEY

/usr/local/bin/start-notebook.sh