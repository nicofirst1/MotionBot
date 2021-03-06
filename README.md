# MotionBot

This project combines visual perception, with Opencv, and telegram bots. The goal is to have a cheap, easy
to use, surveillance system that you can install effortless in your home.

Check out the **new branch**
[darknet security](https://gitlab.com/nicofirst1/MotionBot/tree/darknet_security), use 
```
git checkout darknet-security
```
To move to the new branch.

## Table of Content

- [Behavior](#behavior)
  * [Starting up](#starting-up)
  * [Flags and log](#flags-and-log)
  * [Movement detection](#movement-detection)
  * [Face recognition](#face-recognition)
- [Getting Started](#getting-started)
  * [Package Setup](#package-setup)
  * [Final Setup](#final-setup)
  * [Parameter Tuning](#parameter-tuning)
      - [Cam_movement](#cam-movement)
      - [Cam_shotter](#cam-shotter)
      - [Face_recognizer](#face-recognizer)
- [Usage](#usage)
  * [Avaiable telgram commands](#avaiable-telgram-commands)
  * [Flags](#flags)
- [Author](#author)


## Behavior
In this section I will demonstrate the behavior of the bot.

### Starting up

When you first start the bot it will take an image of what the camera is seeing and save it as a background. You should always start the 
program when there are no moving objects in the camera field of view, the program can programmatically reset your ground image
 but it will take some time.

Here there are some screenshots I captured from my telegram account

![image](https://i.imgur.com/QTtrvGT.png)

As you can see the first image is an actual photo from the camera while the other one is the blurred, grayscaled background image.

As previous mentioned the ground image will programmatically adjusts when some changes in the background are detected, 
like moving chairs, or different light condition. You can force this change to happen with the /resetg command and check the 
current background image with /bkground

![image](https://i.imgur.com/iQey6gc.png)

### Flags and log

You can set various flags within telegram that will change the behavior of the program (for more infos read the *Usage->Flags* section).
To change the flag values just click on the inline button.

![image](https://i.imgur.com/Pz7LkwM.png)

This project make use of a logger, as might be seen the file connected with the logger can be sent and deleted with two simple telegram commands.


### Movement detection

The core of this project is motion detection, that is the ability to determine a movement in the current camera field of view
and notify the user about it.
Depending on the flags value you will get different information from the camera. For example:

When the only flag you set to True is *Motion* you will just get a notification from the camera, if you set bot *Motion* and 
*Video* you will be sent the video of the movement too.
If you set all the flags to True (debug excepted) you will get this:

![image](https://i.imgur.com/Ny34Of9.png)

As you can see, first you have the notification, then the recognized face, Nicolo which is me, (the confidence is just a debug feature 
and will disappear later in the development), and a gif file with a video.

The gif file will have different squares on it, which you will be able to remove with future implementation.
The green ones are for the area of the camera that has been changed, while the red ones are the recognized faces.

Click on the video below to check out the gif file.

[![](http://img.youtube.com/vi/jjpNof2T4MI/0.jpg)](http://www.youtube.com/watch?v=jjpNof2T4MI "Video")

Moreover the date and the movement direction (Incoming/Outgoing , Left/right) will be displayed at the bottom of the gif.

### Face recognition

This project uses a face recognition algorithm [LBPH](https://github.com/informramiz/opencv-face-recognition-python) to guess 
whose face the images belongs to. To do so you must first tell it which face belongs to which person.
Here is where you want to use the /classify command. 

When you first use it this will show up:

![image](https://i.imgur.com/jYTxQ7H.png) 

The **Save Faces** button will let you map faces with names, while the **See Faces** will show all the saved images with the relative name.

By clicking on **Save Faces** you will get the following

![image](https://i.imgur.com/kMirsmy.png)

Since I'm using the desktop version of telegram the button are cut for space problems, so I got the a screenshots from the mobile version

![image](https://i.imgur.com/4XNBQ0M.jpg)

As you can see you have 4 options to choose from.
* The first lines are occupied by saved faces buttons, these are programmatically generated every time you add a new face. By clicking on them
you are telling the bot that THAT specific face belongs to THAT specific person.
* Next you have the **New** button to add a person face. Simply follow the instructions afterwards.
* You may choose to **Delete** the photo if you think it won't be useful to the face recognition (i.e. when the image is blurred, black or even 
not a face)
* Finally you can **Exit** the classification, remember to always do so since that button will trigger the re-training of the recognizer.

If you rather see the saved faces, click on the **See Faces** button and this will show up

![image](https://i.imgur.com/dDHzJfe.png)

Here you can choose, from the saved faces, which you would like to see. The bot will then send you the remaining images from that person

![image](https://i.imgur.com/Du65pvR.png)

**NB** : As mentioned before, every time you save a face and then exit the face recognition model will be retrained. This will
increase the size of the model with approximately 10M every 70 images saved.


## Getting Started

**IMPORTANT**


If you are having any kind of problems related to the following steps check out the *Journal.md* under the *Issues* section. 

### Package Setup
This project is currently working with **raspberry pi 3 model B**, **Python 3.5** and a **Logitech webcam**
* First install the *fswebcam package* (you can check out [this tutorial](https://www.raspberrypi.org/documentation/usage/webcams/)) with 
`sudo apt-get install fswebcam`, control that the cam is working correctly by running `fswebcam image.jpg` (use `eog image.jpg` to view the image through ssh)
* Then follow [this tutorial](https://www.pyimagesearch.com/2016/04/18/install-guide-raspberry-pi-3-raspbian-jessie-opencv-3/)
To install **OpenCV** for raspberry pi (changing python3.4 to python3.5)
* Next install the **scikit-image** package by running `sudo apt-get install python-skimage` followed by `pip install scikit-image` (be sure to be in the correct virtual enviroment using python3.5)
* Install [telegram-python-bot](https://github.com/python-telegram-bot/python-telegram-bot) with `pip install python-telegram-bot --upgrade`
* (Optional) Install profiler fuction `pip install -U memory_profiler`


### Final Setup
* Edit file **token_psw.txt**, insert your token and password after the *=* - the password can be any random string and will be used for authentication against the bot.
* Edit the default_id in *Cam.py* -> *Telegram_handler* -> *__init__*, to your telegram id

### Parameter Tuning
 You may want to tune some parameters depending on your enviroment (light, distance...). Here you will find a complete list of
 the parameter i suggest you to change based on your needs.

#### Cam_movement

You can find the following parameter in the __init__ function

* **send_id** : your telegram id
* **min_area** : the minimum area for the movement detection. If the current frame has a difference with the ground image
and the area of this difference is grater than the **min_area** parameter, the movement is detected
* **frontal_face_cascade/profile_face_cascade** : they must be set to the cascades in the *opencv/data* directory you downloaded
* **max_seconds_retries** (optional) : The movement will be detected for a maximum of *max_seconds_retries* second, then the 
 program will look for background changes
* **resolution** : the resolution you want to use for your camera (Note that if you change this parameter telegram will read the video files
 as Document rather than Gif)
* **fps** : the frame per second for your cam
* **face_photo/motion/debug/video flags** : You can directly run the bot with the default falgs value by setting these parameters (see the flag section below)
* **blur** : the mean by which you want to blur the frames before detecting any movement (use the command /bkground to check the blur ratio)
* **face_size** : the minimum window size to look for faces, the bigger the faster the program gets. But for distant
 people small values are to be taken into account
* **max_blurrines** : the maximum threshold for blurriness detection

#### Cam_shotter

You can find the following parameter in the __init__ function
* **cam_idx** : the index of your camera (it should be zero for one camera)

#### Face_recognizer
* **distance_thres** : The maximum euclidean distance between the frame containing a face and the recognitor label (sort of a confidence)
* **auto_train_dist** : Same as before, but this threshold should be kept low since it deletes images in the unknown directory if the confidence
 is less then than this threshold
* **image_size** : the image size on with execute the trainig and prediction

## Usage

To start the bot simply use
`python main.py`

If you want to run it even when you close the ssh connection use
`nohup python main.py &`



### Avaiable telgram commands

These are the currently available commands for the telegram bot, check out the /help command either in the *handlers.py* file or through the bot itself

* /start - start the bot and provide the password (you get only one chance to type the correct one)
* /photo - get a snapshot from the camera and send it through telegram 
* /video seconds - you can use it with or without the parameter *seconds* and it will send you a gif of form the camera (default duration 5 seconds)
* /flags - you can dis/enable the notification from the movement detection 
* /resetg - reset the ground image in *cam_movement*
* /bkground - send the current background image
* /logsend - send the log file
* /logdel - delete the log file
* /classify : classify the person face
* /help : sends the help 
* Sending a photo will trigger the face recognition which will send you the predicted faces


NB: you can change the commands name just by changing the *CommandHandlers* in the *main.py* file

### Flags
There are currently 4 flags which you can set inside the bot. 
* **Motion Detection** : this flag allow you to be notified when a movement is detected. When enabled you can access the following flags:
  * **Video** : When a movement is detected a video from the camera will be sent as a gif file
  * **Face Photo** : When a movement is detected, the bot will look for faces in the video above and send a face photo (if found)
  * **Face Reco(gnizer)** : When a face is detected from the video, it will try to predict the face name
* **Debug** : When enable you will receive debug image, like the absDifference, thresholding an so on. Note that this slows down the program a lot


## Author

The only Author is [me](https://www.linkedin.com/in/nicol%C3%B2-brandizzi-04091b153/), Nicolò Brandizzi.

I made this project both for pleasure and to exercise my self in writing better README. I made use of some code from the internet, you can 
find everything i used in the *Journal.md* under the *Useful Links* section