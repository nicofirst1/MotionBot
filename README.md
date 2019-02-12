# MotionBot

This project combines visual perception, with Opencv, and telegram bots.
The goal is to have a cheap, easy to use, surveillance system that you can install effortless in your home.


### Darknet Setup
- Follow the instructions in the [yolo wrapper](https://github.com/madhawav/YOLO3-4-Py) to compile the yolo wrapper
- Chang the path of coco.names file in [coco.data](yolo/cfg/coco.data) file from\
`names = data/coco.names`\
to\
 `names= ./src/yolo/data/coco.names`