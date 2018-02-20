# SETUP
## Package Setup
This repo is currently working with **raspberry pi 3 model B** with **Python 3.5** and a **Logitech webcam**
* First install the *fswebcam package* (you can check [this tutorial](https://www.raspberrypi.org/documentation/usage/webcams/)) with 
`sudo apt-get install fswebcam`, check if the cam is working correctly by running `fswebcam image.jpg` (use `eog image.jpg` to view the image throught ssh)
* Then follow [this tutorial](https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)
To install **OpenCV** for raspberry pi (changing python3.4 to python3.5)
* Then install the **scikit-image** package by running `sudo apt-get install python-skimage` followed by `pip install scikit-image` (be sure to be in the correct virtual enviroment using python3.5)
* Install [telegram-python-bot](https://github.com/python-telegram-bot/python-telegram-bot) with `pip install python-telegram-bot --upgrade`
* Install [face_recognition](https://github.com/ageitgey/face_recognition) with `pip3 install face_recognition`
* Install profiler fuction `pip install -U memory_profiler`

## Physical Setup
**THIS PART IS NOT IMPLEMENTED YET**
For this part you need a [microservo motor](https://www.amazon.com/RioRand-micro-Helicopter-Airplane-Controls/dp/B00JJZXRR0)
* Connect it like this

![connection](https://cdn.instructables.com/F6Y/Y4UA/IZT6TFQN/F6YY4UAIZT6TFQN.MEDIUM.jpg)
![connection](https://cdn.instructables.com/F91/2AHG/IZT6TFNU/F912AHGIZT6TFNU.MEDIUM.jpg)

Where the input pin is the GPIO0

![connection](https://cdn.instructables.com/F7X/KHKG/IZT6TIS5/F7XKHKGIZT6TIS5.LARGE.jpg) 

## Final Setup
* Edit file **token_psw.txt**, insert your token and password after the *=*
* Edit the default_id in *Cam.py* -> *Telegram_handler* -> *__init__*, to your telegram id

## Parameter Tuning
 You may want to tune some parameters depending on your enviroment (light, distance...). Here you will find a complete list of
 the parameter i suggest you to change based on your needs.

### Cam_movement

You can find the following parameter in the __init__ function

* **send_id** : your telegram id
* **min_area** : the minimum area for the movement detection. If the current frame has a difference with the ground image
and the area of this difference is grater than the **min_area** parameter, the movement is detected
* **frontal_face_cascade/profile_face_cascade** : they must be set to the cascades in the *opencv/data* direcotry you downloaded
* **max_seconds_retries** (optional) : The movement will be detected for a maximum of **max_seconds_retries** second then, the 
 program will look for background changes
* **resolution** : the resolution you want to use for your camera (Note that if you change this parameter telegram will read the video files
 as Document rather than Gif)
* **fps** : the frame per second for your cam
* **face_photo/motion/debug/video flags** : You can directly run the bot with the default falgs value by setting these parameters (see the flag section below)
* **blur** : the mean by which you want to blur the frames before detecting any movement (use the command /bkground to check the blur ratio)

### Cam_shotter

You can find the following parameter in the __init__ function
* **cam_idx** : the index of your camera (it should be zero for one camera)

### Face_recognizer
* **distance_thres** : The maximum euclidean distance between the frame containing a face and the recognitor label (sort of a confidence)
* **auto_train_dist** : Same as before, but this threshold should be kept low since it deletes images in the unknown direcotry if the confidence
 is less then than this threshold
* **image_size** : the image size on with execute the trainig and prediction

# HOW TO USE

To start the bot simply use
`pyhton main.py`

If you want to run it even when you close the ssh connection use
`nohup python main.py &`



## Avaiable telgram commands

These are the currently avaiable commands for the telegram bot, check out the /help command either in the *handlers.py* file or trhought the bot itself

* /start - strat the bot and provide the password (you get only one chanche to type the correct one)
* /photo - get a snapshot from the camera and send it trhought telegram 
* /video seconds - you can usit with or without the parameter *seconds* and it will send you a gif of form the camera (default duration 5 seconds)
* /flags - you can dis/enable the notification from the movement detection 
* /resetg - reset the ground image in *cam_movement*
* /bkground - send the current background image
* /logsend - send the log file
* /logdel - delete the log file
* /classify : classify the person face
* /help : sends the help 

NB: you can change the commands name just by changing the *CommandHandlers* in the *main.py* file

## Flags
There are currently 4 flags which you can set inside the bot. 
* **Motion Detection** : this flag allow you to be notified when a movement is detected. When enabled you can access the following flags:
  * **Video** : When a movement is detected a video from the camera will be sent as a gif file
  * **Face Photo** : When a movement is detected, the bot will look for faces in the video above and send a face photo (if found)
* **Debug** : When enable you will recieve debug image, like the absDifference, thresholding an so on. Note that this slows down the program a lot

