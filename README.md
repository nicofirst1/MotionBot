# MotionBot

This project combines visual perception, with Opencv, and telegram bots.
The goal is to have a cheap, easy to use, surveillance system that you can install effortless in your home.


### Darknet Setup
- Follow the instructions in the [darknet](https://pjreddie.com/darknet/install/) to compile the darknet environment
- Run  [openimage_downloader](openimage_downloader.sh) in the main directory to download both coco and openimage weights
- If you want to use the coco dataset, change the path of coco.names file in [coco.data](src/darknet/cfg/coco.data) file from\
`names = data/coco.names`\
to\
 `names= ./src/darknet/data/coco.names`
- Else do the same for [coco.data](src/darknet/cfg/openimages.data) file from\
`names = data/openimages.names`\
to\
 `names= ./src/darknet/data/openimages.names`
