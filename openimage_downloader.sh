#!/usr/bin/env bash

cwd=$(pwd)

cd Resources/Weights
echo "downloading weights..."
wget https://pjreddie.com/media/files/yolov3-openimages.weights
wget https://pjreddie.com/media/files/yolov3.weights

cd cwd
cd


echo "downloading cfgs..."

wget https://github.com/pjreddie/darknet/tree/master/cfg/yolov3-openimages.cfg
wget https://github.com/pjreddie/darknet/tree/master/cfg/openimages.data
