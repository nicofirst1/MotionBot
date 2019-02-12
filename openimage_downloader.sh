#!/usr/bin/env bash

cd src/yolo/weights
echo "downloading weights..."
wget https://pjreddie.com/media/files/yolov3-openimages.weights

cd ../cfg

echo "downloading cfgs..."

wget https://github.com/pjreddie/darknet/tree/master/cfg/yolov3-openimages.cfg
wget https://github.com/pjreddie/darknet/tree/master/cfg/openimages.data
