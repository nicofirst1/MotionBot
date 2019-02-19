# Utils

## Commands

Given your raspberry pie IP and username (usr)

- Sync local machine with raspberry pie:
 
 ```
 rsync -avz . urs@IP:/your/path/to/MotionBot
 rsync -avz . pi@192.168.1.4:/home/pi/Work/PycharmProjects/MotionBot

 ```


- To scan your network for the raspberry pi ip use:\
`nmap -sn 192.168.1.0/24`

## Links
- [yolo python wrapper](https://github.com/madhawav/YOLO3-4-Py)
- [darknet page](https://pjreddie.com/darknet/)
- [darknet repo](https://github.com/pjreddie/darknet)
- [face recognition](https://github.com/ageitgey/face_recognition)

# Issues

## Issue1
*Couldn't open file: data/coco.names*

### Solution 
This problem can be solved by changing the path of coco.names file in [coco.data](yolo/cfg/coco.data) file from\
`names = data/coco.names`\
to\
 `names= ./src/darknet/data/coco.names`


## Issue2
*OSError: libdarknet.so: cannot open shared object file: No such file or directory*

### Solution 
In [darknet.py](../src/darknet/python/darknet.py), line 48, change the \
`lib = CDLL("libdarknet.so", RTLD_GLOBAL)`
to\
`lib = CDLL("your/path/to/darknet/libdarknet.so", RTLD_GLOBAL)`

## Issue3
*ERROR:darknet:argument 1: <class 'TypeError'>: wrong type*\
when using dectect method

### Solution 
As written [here](https://github.com/pjreddie/darknet/issues/1384) and [here](https://github.com/pjreddie/darknet/issues/289) do the following:\
- add this lines: 
```
ndarray_image = lib.ndarray_to_image
ndarray_image.argtypes = [POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
ndarray_image.restype = IMAGE
```
before the *classify* function

- add this function before the *classify* function:
```
def nparray_to_image(img): 
    data = img.ctypes.data_as(POINTER(c_ubyte))
    image = ndarray_image(data, img.ctypes.shape, img.ctypes.strides)
    return image
```

- replace:\
` im = load_image(image, 0, 0)` with\
`im = nparray_to_image(image)`

- finally modigy the [image.c](../src/darknet/src/image.c) and [image.h](../src/darknet/src/image.h) as written [here](https://github.com/pjreddie/darknet/issues/289#issuecomment-342448358)  

## Issue4
* No file named data/coco.names*

### Solution
change the path of coco.names file in [coco.data](src/darknet/cfg/coco.data) file from\
`names = data/coco.names`\
to\
 `names= ./src/darknet/data/coco.names`
 
 
## Issue5
```
include/darknet.h:11:30: fatal error: cuda_runtime.h: No such file or directory
     #include "cuda_runtime.h"

```

### Solution 
In the [custom makefile](Resources/Custom_darknet/Makefile) change the use of CUDNN and GPU to zero.


## Issue6

```
     c/_cffi_backend.c:15:17: fatal error: ffi.h: No such file or directory
       #include <ffi.h>
```

### Solution
Install libffi with `sudo apt-get install libffi-dev`


## Issue7
```
ImportError: numpy.core.multiarray failed to import
```

### Solution

Update numpy installation with `ImportError: numpy.core.multiarray failed to import`