# MLOps application stack

## Components

* **S3** exposed at http://localhost:9000
* **Jupyter notebook** exposed at http://localhost:8800
* **MLFlow** exposed at http://localhost:5000
* **Optuna dashboard** exposed at http://localhost:8080
* **PostgreSQL**


## Usage scenario (as intended)

1. Start **ax-ml-ops** application stack with
```
docker-compose up -d --build
```

2. Work on your experiment(s):

  * edit experiment `./services/experiment/experiments/spaceship-titanic/experiment.py` or create a new one under `./services/experiment/experiments/<your-experiment-name>/experiment.py`

  * make some quick insights with **Jupyter**. Take a look at the `sample.ipynb` to see how to import and use artifacts from the experiment. The experiments ale mounted to the `jupyter` container so any changes will be visible instantly (without reloading the container)

3. Run the `experiment` container with
```
./start-experiment.sh
```

4. Monitor results with **MLFlow** and or **Optuna dashboard**

## Todos
* add wait-for-it feature
    * `datastore-s3-config` waits for `datastore-s3`
    * `mlflow` waits for `db` and `datastore-s3`
    * `optuna-dashboard` waits for `db`
    * `experiment` waits for `db` and `datastore-s3` (`mlflow`)
