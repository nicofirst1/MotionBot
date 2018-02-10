#SETUP
This repo is currently working with **raspberry pi 3 model B** with **Python 3.5** and a **Logitech webcam**
* First install the *fswebcam package* (you can check [this tutorial](https://www.raspberrypi.org/documentation/usage/webcams/)) with 
`sudo apt-get install fswebcam`, check if the cam is working correctly by running `fswebcam image.jpg` (use `eog image.jpg` to view the image throught ssh)
* Then follow [this tutorial](https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)
To install **OpenCV** for raspberry pi (changing python3.4 to python3.5)
* Then install the **scikit-image** package by running `sudo apt-get install python-skimage` followed by `pip install scikit-image` (be sure to be in the correct virtual enviroment using python3.5)
* Install [telegram-python-bot](https://github.com/python-telegram-bot/python-telegram-bot) with `pip install python-telegram-bot --upgrade`


# HOW TO USE

## Avaiable telgram commands
* /start - strat the bot and provide the password (you get only one chanche to type the correct one)
* /photo - get a snapshot from the camera and send it trhought telegram 
* /video seconds - you can usit with or without the parameter *seconds* and it will send you a gif of form the camera (default duration 5 seconds)
* /notification - you can dis/enable the notification from the movement detection part




# USEFUL LINKS
## Opencv 
### API
* https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
* https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video

### CAPTURE VIDEO
* http://answers.opencv.org/question/128081/python-frame-grabbing-from-ip-camera-and-process-in-a-different-thread/

### Image similarity comparison
* https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf

## With telegram
* https://github.com/meinside/telegram-bot-opencv


# TODO

## Raspberry
- [ ] cron job to update when gitlab difference detected
- [ ] cron job to start the bot at 8 

## Code
- [X] fix mp4 video on telegram mobile
- [ ] Command to stop bot execution
- [ ] Get token and psw from file
- [ ] Get Classifier path from home direcotry
- [ ] Save images/videos with format *video-user_id.extension*

### Movement detection
- [X] Nofity when movement is detected 
- [X] Enable/disable notification
- [X] Send different image
- [X] Send different video
- [X] Detect face in image change
- [ ] Draw rectangle around face

# Issues

## Telegram gif not showing up on mobile
### Issue
Using 
> codec= cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(video_name, codec, fps,(640,480))
out.write(frame)


Generate a .mp4 video with is shown as a *gif* in telegram. While the desktop version has no problem viewing it the mobile version 
displays a blank file wich can be seen only by downloading the .mp4.

While generating the file *OpenCv* yelds the following warning
> OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported with codec id 13 and format 'mp4 / MP4 (MPEG-4 Part 14)'
OpenCV: FFMPEG: fallback to use tag 0x00000020/' ???'


### Tried Fixes
* Changing the resolution from *640,480* to any other resolution brings telegram to recognize the file as a video (not gif), but it still does not show up in the mobile version
* Changing the file extension to *.mp4v* does not work 
* Changing codec to cv2.VideoWriter_fourcc(*'MPEG') does not show gif on desktop too
* Using **isColor=False** does not workù

### Solutions
* Changing codec to _cv2.VideoWriter_fourcc(*'avc1')_ and extension to **.mov** sends a file (not a gif) which can be viewd both by the desktop and the mobile version of telegram
* Final solution: Removed the codec calss and used **0x00000021** instead (with **.mp4** extension), found (here)[https://devtalk.nvidia.com/default/topic/1029451/-python-what-is-the-four-characters-fourcc-code-for-mp4-encoding-on-tx2/]

## Video difference is laggy 
### Issue
The video difference is send when a difference in frame is detected, this detection is time costly thus writing a frame to the video object too slowly.
This brings to a laggy gif file

### Tried Fixes
* Remove *sleep(1/self.fps)* from while loop...not working
* Remove face detection...
