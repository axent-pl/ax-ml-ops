# axent.pl MLOps

## Usage scenario (as intended)

1. Start **ax-ml-ops** application stack with
```
docker-compose up -d --build
```
2. Adjust the provided experiment `./services/experiment/experiments/spaceship-titanic/experiment.py` or create a new one under `./services/experiment/experiments/<your-experiment-name>/experiment.py`
3. When needed make some quick insights with **Jupyter**. Take a look at the `sample.ipynb` to see how to import and use artifacts from the experiment.
4. Whenever required run the `experiment` container once again with
```
docker-compose stop experiment
docker-compose up -d --build
```
5. Monitor results with mlflow and or optuna dashboard

## Todos
* add wait-for-it feature
    * `datastore-s3-config` waits for `datastore-s3`
    * `mlflow` waits for `db` and `datastore-s3`
    * `optuna-dashboard` waits for `db`
    * `experiment` waits for `db` and `datastore-s3` (`mlflow`)
