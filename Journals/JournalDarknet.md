# Utils

## Commands
- Sync local machine with raspberry pie:\
`rsync -avz . pi@192.168.1.4:/home/pi/Work/PycharmProjects/MotionBot`

- To scan your network for the raspberry pi ip use:\
`nmap -sn 192.168.1.0/24`

## Links
- [yolo python wrapper](https://github.com/madhawav/YOLO3-4-Py)
- [darknet page](https://pjreddie.com/darknet/)
- [darknet repo](https://github.com/pjreddie/darknet)

# Issues

## Issue1
Couldn't open file: data/coco.names

### Solution 
This problem can be solved by changing the path of coco.names file in [coco.data](yolo/cfg/coco.data) file from\
`names = data/coco.names`\
to\
 `names= ./src/yolo/data/coco.names`


## Issue2
OSError: libdarknet.so: cannot open shared object file: No such file or directory

### Solution 
In [darknet.py](../src/darknet/python/darknet.py), line 48, change the \
`lib = CDLL("libdarknet.so", RTLD_GLOBAL)`
to\
`lib = CDLL("your/path/to/darknet/libdarknet.so", RTLD_GLOBAL)`
