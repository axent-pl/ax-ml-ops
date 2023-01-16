FROM jupyter/datascience-notebook:python-3.10

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY sample.ipynb /home/jovyan/

USER root