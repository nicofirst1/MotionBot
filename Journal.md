

# USEFUL LINKS

## Opencv 

### API
* https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
* https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#display-video

### CAPTURE VIDEO
* http://answers.opencv.org/question/128081/python-frame-grabbing-from-ip-camera-and-process-in-a-different-thread/

### Motion detection
* https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

### Face recognition
* https://github.com/informramiz/opencv-face-recognition-python
* https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_api.html#Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius, int neighbors, int grid_x, int grid_y, double threshold)
* https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_api.html

## Image similarity comparison
* https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf

## With telegram
* https://github.com/meinside/telegram-bot-opencv

## Servo
* https://learn.adafruit.com/adafruits-raspberry-pi-lesson-8-using-a-servo-motor/software

# TODO

## Raspberry
- [ ] cron job to start the bot at 8 

## General
- [X] Get token and psw from file
- [ ] ~~Get Classifier path from home direcotry~~ 
- [X] Save images/videos with format *video-user_id.extension*
- [X] use Cam_shotter to get video
- [X] Stop/start cam_motion class by flag value
- [X] reorganize prints
- [X] implement a logger
- [ ] Add error handling at the origin to not stop the class
- [X] Fix logging , do not print on terminal
- [ ] Comment code 
- [X] Add requirements.txt
- [ ] Forgiveness instead of Permission

## Telegram
- [X] fix mp4 video on telegram mobile
- [X] Command to stop bot execution
- [X] Make custom inline keyboard to set flags
- [X] User friendly motion detection notification
- [X] Send caption with image
- [X] Command to reset ground image
- [X] Reset ground image -> stops motion tasks
- [X] Add command to send background image
- [X] Fix send background command
- [X] Fix reset background command
- [ ] Surround with try/except every bot_edit_message for the error *telegram.error.BadRequest: Message is not modified*
- [X] Write help command
- [X] Add help command 
- [X] Fix video command

## Camera movement
- [ ] Use step motor with GPIO to move the camera
- [ ] Take a video while the camera performs a 180° rotation
- [ ] Integrate movement with background reset

## Movement detection
- [X] Nofity when movement is detected 
- [X] Enable/disable notification
- [X] Send different image
- [X] Send different video
- [X] Detect face in image change
- [X] Draw rectangle around face
- [X] Find something faster than SSIM -> MSE
- [X] Get face photo
- [X] Denoise photo
- [X] Wait after cam is opened
- [X] Add date time to difference video
- [X] Remove rectangles from face recognition
- [X] Add profile face detector
- [X] Fix are_different loop
- [ ] Fix date display
- [X] Reset ground image programmaticly
- [X] detect movement direction (right,left) (position of areas) 
- [X] detect movement direction (incoming, outcoming) (sum of areas)
- [ ] detect multiple faces
- [X] Update motion notifier

## Face Recognizer
- [X] Save faces into corresponding dirs
- [X] Train model for every face
- [X] Classify person
- [X] Find a goddamn way to get the classification confidence
- [X] Resize all training/predition images to same size
- [X] Save model 
- [X] Load/update model with new faces
- [X] Delete faces which have been updated into the recognitor
- [X] Get face confidence
- [X] Delete unkown faces classified with a confidence < 70
- [ ] Send photo and recognize faces in image

## Optimization
- [X] New thread class for image/video/message sending
- [X] Fix while score, exit when no difference are detected anymore
- [X] Save countour list
- [X] Save area list
- [X] Implement profiling function
- [X] Optimize are different (replace for with any)
- [X] Moved from gaussianBlur to blur (x4 times faster)
- [ ] Optimize face recognition 
- [X] Optimize face detection in time (detectMultiScale is slow)
- [X] Delete subjects face images after the model has been trained with them
- [ ] Saving the recognizer object create a yaml file of 17M, while the photo in the Faces direcories are 4M...
check out if the yaml file increses or stays constant in size

- [ ] ~~New thread function to get face in video~~

# Updates

* Add possibility to send infos to specific id in telegram class
* Updated README
* Add class Face_recognizer which allow to save the images of faces with the corresponding name
* Add flag for face recognition
* Finally got the face recognition confidence
* Auto train for the recognizer and unknown images, with the update method
* Save/Load the recognizer from yaml file
* Removed detectMultiScale and replace it with multiple face prediction to get the best faces and  faster prediction
* Optimized code now it is 27% faster

# Issues

## Issue
Telegram gif not showing up on mobile
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
* Changing codec to _cv2.VideoWriter_fourcc(*'MPEG')_ does not show gif on desktop too
* Using **isColor=False** does not work

### Solutions
* Changing codec to _cv2.VideoWriter_fourcc(*'avc1')_ and extension to **.mov** sends a file (not a gif) which can be viewd both by the desktop and the mobile version of telegram
* Final solution: Removed the codec calss and used **0x00000021** instead (with **.mp4** extension), found [here](https://devtalk.nvidia.com/default/topic/1029451/-python-what-is-the-four-characters-fourcc-code-for-mp4-encoding-on-tx2/)

## Issue
Video difference is laggy 
The video difference is send when a difference in frame is detected, this detection is time costly thus writing a frame to the video object too slowly.
This brings to a laggy gif file.
GRAY SCALING takes 0.01 seconds
SSIM takes about 0.5 seconds for every image, while gray scale takes 0.01 seconds
PSNR takes 0.04 seconds for every image

### Tried Fixes
* Remove *sleep(1/self.fps)* from while loop...not working
* Remove face detection...not working

### Solution
* Taking the frames in the Cam_shotter class resolved the issue

## Issue 
If you are having an error like:
> VIDEOIO ERROR: V4L: index 0 is not correct!

Change the **cam_idx** in Cam_shotter to the correct one for your raspberry pi

## Warning

Encountered when the cam_movement class first start to compute difference between images
>python3.5/site-packages/skimage/measure/simple_metrics.py:142: RuntimeWarning: divide by zero encountered in double_scalars
  return 10 * np.log10((data_range ** 2) / err)

When the cam_shotter class compl

## Issue
Using the *haarcascades/haarcascade_frontalface_alt.xml* with *CascadeClassifier* yelds a great number of false-positive

### Solution 
Changing to *haarcascades/haarcascade_frontalface_alt_tree.xml* resolved the issue

## Issue 
Found error while performing the abs difference *frameDelta = cv2.absdiff(grd_truth, gray)*. In cam_movement class
> OpenCV Error: Sizes of input arguments do not match (The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array') in arithm_op, file /home/pi/InstallationPackages/opencv-3.1.0/modules/core/src/arithm.cpp, line 639
Cv Error: /home/pi/InstallationPackages/opencv-3.1.0/modules/core/src/arithm.cpp:639: error: (-209) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function arithm_op

* It seems to be correlated to the number of channels of the images passed.
* When the error occurres the grd_thruth  shape is (480, 640, 3) while the gray is (480,640), the number 3 should not be there since the image is being converted to gray scale with
> gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

### Tried Fixes
* Sorround difference with try catch

### Solution
* Forgot to call  cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) XD

## Issue
If you recieve the following message when starting the program with `python main.py`:
>libv4l2: error setting pixformat: Device or resource busy
VIDEOIO ERROR: libv4l unable to ioctl S_FMT
libv4l2: error setting pixformat: Device or resource busy
libv4l1: error setting pixformat: Device or resource busy
VIDEOIO ERROR: libv4l unable to ioctl VIDIOCSPICT

Use `killall pyhton` (This will stop every pyhton process currently running)

## Issue
Telegram command to get the ground image of the Cam_movement calss seems to stop while writing the image to file. It may be
connected with the continuous use of the ground_image inside the movement class.
It is connected only to the cv.imwrite() function

### Tried Fixes

* Implement a get method
* Return a copy of the object

### Solution
* Send image throught cam_movement class

## Issue
Get the prediction confidence with the *cv2.face.createLBPHFaceRecognizer().predict()* method

### Tried Fixes
* Followed [this](http://answers.opencv.org/question/82294/cant-get-predict-confidence/), but no luck

### Solved
solved by using the collector object
 
 



# Optimization

## Image difference


| Algorithm        | Time taken in seconds| Suggested range |
| -------------    |:-------------:       |  -----:         |
| GRAY SCALING     | 0.01                 |                 |
| SSIM             | 0.5                  | x  <   0.75     |
| PSNR             | 0.03                 | x  <   30       |
| NRMSE            | 0.035                | x  >   0.3      |
| MSE              | 0.025                | x  > 500       |


### MSE
* Change in shadow with value 3919
* It does not detect image far away persons
* Switched to PSNR

### PSNR
* Way more sensible than MSE (in a good way)
* Not so sensitive to shadow changes
* Change detected with score 24, while there was none 
* Is triggered when camera auto adjust brightness
* In bright places it becomes very sensitive -> the use of an *equalizeHist* seems to resolve the problem
* No good in poor light condition

### SSIM
* Using gaussian_weights=True -> time increases to 0.7 seconds

### Passing to  cv2.absdiff

## detectMultiScale

* Currently detectMultiScale is the slowest part of the program, it takes up to 30 seconds fo detect an image. With a time per call of 0.065
* I'm using  scale_factor = 1.4 and min_neight = 3.
* Setting min_size fto (20,20) doesn't change anything
* Setting the min_size to (50,50) speeds up the computation by x3
* Setting mi_size to (100,100) ... small faces won't be recognized
* Setting the min_size to (75,75) too big ... keeping 50

A solution could be paralleling the function for all the frames

## cvtColor
* Taking up to 12% of total time, per call time is   0.013. It is done twice for every frame

A solution could be using the cvtColor inside the cam_shotter, for every first frame.

## face_FaceRecognizer.predict
* Taking up 10% of time with 0.117 seconds per call.
* Try to use paralleling programming