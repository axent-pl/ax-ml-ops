FROM jupyter/datascience-notebook:python-3.9.13

COPY requirements.txt /
RUN pip install -r /requirements.txt

# COPY sample.ipynb /home/jovyan/
COPY configure-and-start-notebook.sh /usr/local/bin/

USER root
CMD [ "configure-and-start-notebook.sh" ]