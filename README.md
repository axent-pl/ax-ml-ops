# ax-ml-ops

## Todos
* add wait-for-it feature
    * `datastore-s3-config` waits for `datastore-s3`
    * `mlflow` waits for `db` and `datastore-s3`
    * `optuna-dashboard` waits for `db`
    * `experiment` waits for `db` and `datastore-s3` (`mlflow`)
* add devpi server https://devpi.net/docs/devpi/devpi/latest/+doc/index.html
* add jupyter notebook with axent package
* add axent package to devpi server so it can be used by experiments and a jupyter notebook
* unify and govern experiment structure so it can be easier transfered between jupyter and the experiment container
