export MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlops:${DB_MLOPS_PASS}@db/mlops
export MLFLOW_S3_ENDPOINT_URL=http://datastore-s3:9000
export AWS_ACCESS_KEY_ID=$S3_AUTH_KEY_ID
export AWS_SECRET_ACCESS_KEY=$S3_AUTH_SECRET_KEY

mlflow server -h 0.0.0.0 --backend-store-uri postgresql+psycopg2://mlops:${DB_MLOPS_PASS}@db/mlops --default-artifact-root s3://${S3_BUCKET}/ --artifacts-destination s3://${S3_BUCKET}/