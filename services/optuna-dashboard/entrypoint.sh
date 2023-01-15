#!/bin/bash
optuna-dashboard --port 8080 --host 0.0.0.0 --server gunicorn postgresql+psycopg2://optuna:${DB_OPTUNA_PASS}@db/optuna
