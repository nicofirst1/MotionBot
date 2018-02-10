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
- [ ] fix mp4 video on telegram mobile
- [ ] Command to stop bot execution
- [ ] Get token and psw from file

### Movement detection
- [X] Nofity when movement is detected 
- [X] Enable/disable notification
- [X] Send different image
- [ ] Send different video
- [X] Detect face in image change
- [ ] Draw rectangle around face