export MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://${DB_MLFLOW_USER}:${DB_MLFLOW_PASS}@${DB_HOST}/${DB_MLFLOW_NAME}
export MLFLOW_S3_ENDPOINT_URL=http://${S3_HOST}:${S3_PORT}
export MLFLOW_ARTIFACTS_DESTINATION=s3://${S3_BUCKET}/
export AWS_ACCESS_KEY_ID=$S3_AUTH_KEY_ID
export AWS_SECRET_ACCESS_KEY=$S3_AUTH_SECRET_KEY

mlflow server -h 0.0.0.0 --backend-store-uri postgresql+psycopg2://${DB_MLFLOW_USER}:${DB_MLFLOW_PASS}@${DB_HOST}/${DB_MLFLOW_NAME} --default-artifact-root s3://${S3_BUCKET}/ --artifacts-destination s3://${S3_BUCKET}/