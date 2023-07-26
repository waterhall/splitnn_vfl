#!/bin/bash
export MLFLOW_S3_ENDPOINT_URL=http://34.67.88.74;
export AWS_ACCESS_KEY_ID=miniomlflow;
export AWS_SECRET_ACCESS_KEY=R9RqzmC1;
export MLFLOW_TRACKING_URI=http://34.67.88.74:5000;

for RNG in 0 1 2
do
    mlflow run . -e pytorch -P config_file=mnist_une_local.yaml -P random_seed=$RNG
    sleep 30
    mlflow run . -e framework -P config_file=mnist_une_local.yaml -P random_seed=$RNG
    sleep 30
    python start_clients.py mnist_une.yaml $RNG
    sleep 30
    python start_clients.py mnist_smpc.yaml $RNG
    sleep 30
    python start_clients.py mnist_pe.yaml $RNG
    sleep 30
    python start_clients.py mnist_tpe.yaml $RNG
    sleep 30
done

