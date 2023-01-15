# FROM ghcr.io/mlflow/mlflow:v2.1.1
# RUN pip install psycopg2-binary
FROM python:3.9.13

COPY requirements.txt /
RUN pip install -r requirements.txt

COPY entrypoint.sh /

WORKDIR /
ENTRYPOINT "./entrypoint.sh"