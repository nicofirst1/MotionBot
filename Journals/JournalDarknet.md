# Utils

## Commands
- Sync local machine with raspberry pie:\
`rsync -avz . pi@192.168.1.4:/home/pi/Work/PycharmProjects/MotionBot`

- To scan your network for the raspberry pi ip use:\
`nmap -sn 192.168.1.0/24`


# Issues

## Issue1
Couldn't open file: data/coco.names

### Solution 
This problem can be solved by changing the path of coco.names file in [coco.data](yolo/cfg/coco.data) file from\
`names = data/coco.names`\
to\
 `names= ./src/yolo/data/coco.names`
