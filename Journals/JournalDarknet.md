# Utils

## Commands
- Sync local machine with raspberry pie:\
`rsync -avz . pi@192.168.1.4:/home/pi/Work/PycharmProjects/MotionBot`

- To scan your network for the raspberry pi ip use:\
`nmap -sn 192.168.1.0/24`

## Links
- [yolo python wrapper](https://github.com/madhawav/YOLO3-4-Py)
- [darknet page](https://pjreddie.com/darknet/)

# Issues

## Issue1
Couldn't open file: data/coco.names

### Solution 
This problem can be solved by changing the path of coco.names file in [coco.data](yolo/cfg/coco.data) file from\
`names = data/coco.names`\
to\
 `names= ./src/yolo/data/coco.names`
