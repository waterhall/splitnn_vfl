#!/bin/bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000;
#python plot_ua_bytes.py
#python plot_ua_time.py
#python plot_ccfraud.py

export MLFLOW_TRACKING_URI=http://34.67.88.74:5000/;
python plot_mnist.py
