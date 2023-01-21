FROM python:3.9.13

ARG UID
ARG GID
RUN groupadd -g ${GID} airflow && \
    useradd -u ${UID} -g ${GID} -G sudo -m airflow

USER airflow
WORKDIR /home/airflow
ENV AIRFLOW_HOME=/home/airflow

COPY --chown=airflow:airflow requirements.txt /home/airflow/
RUN pip install -r requirements.txt

COPY --chown=airflow:airflow entrypoint.sh /home/airflow/

ENTRYPOINT [ "/home/airflow/entrypoint.sh" ]