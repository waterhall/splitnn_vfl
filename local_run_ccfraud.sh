#!/bin/bash
for RNG in 1 2
do
    #mlflow run . -e pytorch -P config_file=ccfraud_une_local.yaml -P random_seed=$RNG
    #mlflow run . -e pytorch -P config_file=ccfraud_une_partial_local.yaml -P random_seed=$RNG
    mlflow run . -e pytorch -P config_file=ccfraud_une_mid_partial_local.yaml -P random_seed=$RNG
    #mlflow run . -e pytorch -P config_file=ccfraud_une_worst_partial_local.yaml -P random_seed=$RNG
    sleep 10
done

