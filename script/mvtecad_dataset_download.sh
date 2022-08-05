#!/bin/bash

mkdir -p ./data
mkdir -p ./data/mvtec_ad/

wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz -O ./data/mvtec_anomaly_detection.tar.xz
tar -Jxvf ./data/mvtec_anomaly_detection.tar.xz -C ./data/mvtec_ad