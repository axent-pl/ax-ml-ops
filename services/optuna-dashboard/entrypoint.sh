#!/bin/bash
optuna-dashboard --port 8080 --host 0.0.0.0 --server gunicorn postgresql+psycopg2://${DB_OPTUNA_USER}:${DB_OPTUNA_PASS}@${DB_HOST}/${DB_OPTUNA_NAME}
