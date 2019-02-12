#!/usr/bin/env bash

cwd=$(pwd)
# create dir if does not exists
mkdir -p ./Resources/Weights


cd Resources/Weights

if [ ! -f yolov3-openimages.weights ]; then
    echo "downloading openimage weights..."
    wget https://pjreddie.com/media/files/yolov3-openimages.weights

fi
if [ ! -f yolov3-openimages.weights ]; then
    echo "downloading coco weights..."
    wget https://pjreddie.com/media/files/yolov3.weights

fi

cd $cwd

echo "copying custom files"

cp ./Resources/Custom_darknet/coco.data ./src/darknet/cfg/coco.data
cp ./Resources/Custom_darknet/openimages.data ./src/darknet/cfg/openimages.data

cp ./Resources/Custom_darknet/image.c ./src/darknet/src/image.c
cp ./Resources/Custom_darknet/image.h ./src/darknet/src/image.h

cp ./Resources/Custom_darknet/Makefile   ./src/darknet/Makefile

cp ./Resources/Custom_darknet/darknet.py   ./src/darknet/python/darknet.py


cd ./src/darknet/

make
