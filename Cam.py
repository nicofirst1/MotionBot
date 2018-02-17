import copy
import threading
import traceback
from multiprocessing import Pool
from threading import Thread
import os
import cv2
from time import sleep
import datetime
import logging

import gc

import sys
from memory_profiler import profile

from Face_recognizer import Face_recognizer
from utils import time_profiler

logger = logging.getLogger('motionlog')


class Cam_class:

    def __init__(self, updater):

        self.frames = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.updater=updater
        self.disp=updater.dispatcher
        self.bot=self.disp.bot

        self.telegram_handler = Telegram_handler(  self.bot)
        self.telegram_handler.start()

        self.shotter = Cam_shotter(self.frames)
        self.shotter.start()

        self.face_recognizer=Face_recognizer(self.disp)
        self.face_recognizer.start()

        self.motion = Cam_movement(self.shotter, self.telegram_handler, self.face_recognizer)
        self.motion.start()
        logger.debug("Cam_class started")

    def capture_image(self, image_name):
        # print("taking image")
        img = self.frames[-2]

        if isinstance(img, int):
            print("empty queue")
            return False
        # try to save the image
        return cv2.imwrite(image_name, img)



    def capture_video(self, video_name, seconds):
        # set camera resolution, fps and codec
        frame_width = 640
        frame_height = 480
        fps = 20
        # print("initializing writer")

        out = cv2.VideoWriter(video_name, 0x00000021, fps, (frame_width, frame_height))

        # print("writer initialized")

        # start capturing frames
        self.shotter.capture(True)
        # sleep
        sleep(seconds)
        # get catured frames
        to_write = self.shotter.capture(False)
        # write frame to file and release
        for elem in to_write:
            out.write(elem)
        out.release()

        self.telegram_handler.send_video(video_name,str(seconds)+" seconds record")


class Cam_shotter(Thread):
    """Class to take frames from camera"""

    def __init__(self, queue):

        # init the thread
        Thread.__init__(self)

        # get camera and queue
        self.cam_idx = 0
        self.CAM = cv2.VideoCapture(self.cam_idx)
        self.queue = queue
        self.capture_bool = False
        self.capture_queue = []
        self.lock = threading.Lock()
        self.camera_connected = False

        logger.debug("Cam_shotter started")

    def run(self):
        """Main thread loop"""

        while True:

            # read frame form camera
            ret, img = self.CAM.read()

            # if frame has been read correctly add it to the end of the list
            if ret:
                # if it is the first time that the class reads an image
                if not self.camera_connected:
                    print("camera connected")
                    logger.debug("Camera connected")
                    self.camera_connected = True
                    # sleep to wait for auto-focus/brightness
                    sleep(3)
                # pop first element
                self.queue.pop(0)
                # append image at last
                self.queue.append(img)
                if self.capture_bool:
                    self.capture_queue.append(img)
                # print("saved")
            else:
                # try to reopen the camera
                # print("not saved")
                self.reopen_cam()

            # sleep(0.01)

    def capture(self, capture):

        try:
            if capture:
                self.lock.acquire()
                self.capture_queue = []
                self.capture_bool = True
            else:
                self.capture_bool = False
                self.lock.release()
                return self.capture_queue
        except:
            self.lock.release()

    def reopen_cam(self):
        """Function to reopen the camera"""
        # print("reopening cam")
        # release the camera
        self.CAM.release()
        sleep(2)
        # capture stream
        self.CAM = cv2.VideoCapture(self.cam_idx)
        sleep(2)
        # chech if camera is opened
        self.check_open_cam()

    def close_cam(self):
        """Function to release teh camera stream"""
        # print("close cam")
        self.CAM.release()

    def check_open_cam(self):
        """Function to open the camera stream"""
        # print("checking cam")
        if not self.CAM.isOpened():
            # print("cam was closed")
            self.CAM.open(0)


class Cam_movement(Thread):
    """Class to detect movement from camera frames"""

    def __init__(self, shotter, telegram, face_recognizer):
        # init the thread
        Thread.__init__(self)

        self.shotter = shotter
        self.frame = shotter.queue
        self.telegram_handler = telegram
        self.send_id = 24978334
        self.face_recognizer=face_recognizer

        self.delay = 0.1
        self.image_name = "different.png"
        self.min_area = 2000
        self.ground_frame = 0

        self.frontal_face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/lbpcascades/lbpcascade_frontalface.xml')


        self.profile_face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/lbpcascades/lbpcascade_profileface.xml')

        self.max_seconds_retries = 10

        self.video_name = "detect_motion_video.mp4"
        self.resolution = (640, 480)  # width,height
        self.fps = 30

        self.video_flag = True
        self.face_photo_flag = True
        self.motion_flag = True
        self.debug_flag = False

        self.resetting_ground = False

        self.blur=(10,10)

        logger.debug("Cam_movement started")

    def run(self):

        # wait for cam shotter to start
        while not self.shotter.camera_connected:
            sleep(0.5)

        # wait for the frame queue to be full
        initial_frame = self.frame[-1]
        while isinstance(initial_frame, int):
            initial_frame = self.frame[-1]

        # get the background image and save it
        self.reset_ground("Background image")

        while True:
            try:
                self.detect_motion_video()
            except :
                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                logger.error(''.join('!! ' + line for line in lines))  # Log it or whatever here

    def detect_motion_video(self):

        # whait for resetting to be over
        while self.resetting_ground:
            continue

        # get end frame after delay seconds
        sleep(self.delay)
        end_frame = self.frame[-1]

        # calculate diversity
        score = self.are_different(self.ground_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames
        if self.motion_flag and score and not self.resetting_ground:

            logger.info("Movement detected")
            # send message
            self.motion_notifier(score)

            # do not capture video nor photo, just notification
            if not self.video_flag:
                sleep(3)
                return


            # start saving the frames
            self.shotter.capture(True)

            # self.send_image(initial_frame,"initial frame")
            # self.send_image(end_frame,"end frame")

            # while the current frame and the initial one are different (aka some movement detected)
            self.loop_difference(score, self.ground_frame, self.max_seconds_retries)

            # save the taken frames
            to_write = self.shotter.capture(False)

            # if the user wants the video of the movement
            if self.face_photo_flag:
                # take the face and send it
                face = self.face_from_video(to_write)

                if len(face)==0:
                    self.telegram_handler.send_message(msg="Face not found")
                else:
                    self.telegram_handler.send_image(face, msg="Face found")


            # send the original video too
            if not self.resetting_ground:
                self.draw_on_frames(to_write)
                self.telegram_handler.send_video(self.video_name)

            #sleep(3)


    # =========================Movement=======================================

    def check_bk_changes(self, initial_frame, seconds):
        #taking time
        start = datetime.datetime.now()
        end = datetime.datetime.now()
        #setting initial variables
        score = 0

        #setting initial frame
        gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray = cv2.blur(gray, self.blur, 0)

        while not score:

            # take another frame
            prov = self.frame[-1]

            # check if images are different
            score = self.are_different(gray, prov)

            # if time is exceeded exit while
            if (end - start).seconds > seconds:
                print("max seconds exceeded")
                self.reset_ground("Back ground changed! New background")
                return True

            # update current time in while loop
            end = datetime.datetime.now()

        return False

    def loop_difference(self, initial_score, initial_frame, seconds,retry=False):
        """Loop until the current frame is the same as the ground image or time is exceeded"""

        if retry:print("retriyng")
        start = datetime.datetime.now()
        end = datetime.datetime.now()
        score = initial_score
        print("Start of difference loop")
        while score and not self.resetting_ground:

            # take another fram
            prov = self.frame[-1]

            # check if images are different
            score = self.are_different(initial_frame, prov)

            # if time is exceeded exit while
            if (end - start).seconds > seconds:
                print("max seconds exceeded...checking for background changes")
                if not retry: self.check_bk_changes(prov, 3)
                print("End of difference loop")
                return

            # update current time in while loop
            end = datetime.datetime.now()
            sleep(0.05)

        #it may be that there is no apparent motion
        if not retry:
            #wait a little and retry
            sleep(0.5)
            self.loop_difference(1, initial_frame, 1, True)

        print("End of difference loop")

    def are_different(self, grd_truth, img2):

        cnts=self.compute_img_difference(grd_truth,img2)

        return any(cv2.contourArea(elem) > self.min_area for elem in cnts)

    def compute_img_difference(self,grd_truth, img2):
        # print("Calculation image difference")

        # blur and convert to grayscale
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray = cv2.blur(gray, self.blur, 0)

        # print(gray.shape, grd_truth.shape)

        # compute the absolute difference between the current frame and
        # first frame
        try:
            frameDelta = cv2.absdiff(grd_truth, gray)
        except cv2.error as e:
            error_log = "Cv Error: " + str(e) + "\ngrd_thruth : " + str(grd_truth.shape) + ", gray : " + str(
                gray.shape) + "\n"
            logger.error(error_log)
            print(error_log)
            return True

        thresh_original = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]

        # self.send_image(frameDelta,"frameDelta")
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh_original, None, iterations=5)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.debug_flag:
            self.telegram_handler.send_image(frameDelta)
            self.telegram_handler.send_image(thresh_original, msg="Threshold Original")
            self.telegram_handler.send_image(thresh, msg="Threshold Dilated")
            self.telegram_handler.send_image(img2)

        return cnts

    # =========================UTILS=======================================
    def draw_on_frames(self, frames,areas=True, date=True):
        """Function to draw squares on objects"""

        face_color=(0,0,255) #red
        motion_color=(0,255,0) #green
        line_tickness=2

        # create the file
        out= cv2.VideoWriter(self.video_name, 0x00000021, self.fps, self.resolution)
        out.open(self.video_name, 0x00000021, self.fps, self.resolution)

        prov_cnts=0
        idx=0
        to_write = "Unkown - Unkown"
        for frame in frames:


            if self.face_photo_flag:

                # detect if there is a face
                face = self.detect_face(frame)

                # if there is a face
                if len(face) > 0:
                    # get the corners of the faces
                    for (x, y, w, h) in face:
                        # draw a rectangle around the corners
                        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, line_tickness)

            # draw movement
            if areas:
                cnts = self.compute_img_difference(self.ground_frame, frame)



                # draw contours
                for c in cnts:
                    # if the contour is too small, ignore it
                    # print("Area : "+str(cv2.contourArea(c)))
                    if cv2.contourArea(c) < self.min_area:
                        pass

                    else:
                        # compute the bounding box for the contour, draw it on the frame,
                        # and update the text
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), motion_color, line_tickness)


                # add black rectangle at the bottom
                cv2.rectangle(frame, (0, frame.shape[0]), (frame.shape[1], frame.shape[0] - 30), (0, 0, 0), -1)

                # write the movement direction
                if not idx%2:prov_cnts=cnts
                elif len(prov_cnts)>0 and len(cnts)>0:
                    movement,_=self.movement_direction(prov_cnts,cnts)
                    to_write=""
                    if movement[0]:
                        to_write+="Incoming - "

                    else:
                        to_write+="Outgoing - "

                    if movement[1]:
                        to_write += "Left"

                    else:
                        to_write += "Right"


                    #cv2.circle(frame, center_points[0], 1, (255, 255, 0), 10,2)
                    #cv2.circle(frame, center_points[1], 1, (255, 255, 255), 10,2)

                cv2.putText(frame, to_write,
                            (frame.shape[1] - 250, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255), 1)



                idx+=1

            #add a date to the frame
            if date:

                # write time
                correct_date= datetime.datetime.now() + datetime.timedelta(hours=1)

                cv2.putText(frame, correct_date.strftime("%A %d %B %Y %H:%M:%S"),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)


            #write frames on file
            out.write(frame)


        # free file
        out.release()


    def reset_ground(self, msg):
        """function to reset the ground truth image"""
        print("Reset ground image ...")
        self.resetting_ground = True
        gray = cv2.cvtColor(self.frame[-1], cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray = cv2.blur(gray, self.blur, 0)
        self.ground_frame = gray
        self.telegram_handler.send_image(self.ground_frame, msg=msg)
        self.resetting_ground = False
        print("Done")

    def movement_direction(self, cnts1, cnts2):
        """Function to get the movement direction from two frames
        @:return: tuple where the first elem is the outgoing/incoming 0/1, the second is right/left 0/1"""

        #the incoming outgoing movement is estimated throught the sum of the areas
        # if sum(areas1)<sum(areas2) the object is outgoing

        area1=sum(cv2.contourArea(c) for c in cnts1)
        area2=sum(cv2.contourArea(c) for c in cnts2)

        #print(area1,area2)

        #the left/right is given by the position of the averaged center of each area
        # if center1>center2 the object is moving right to left (so left)
        #get the centers of each area on the x axis
        centers1=[]
        centers2=[]

        for c in cnts1:
            centers1.append(cv2.boundingRect(c))

        center_point1=[((x + (x+w))/2,(y + (y+h))/2)  for (x,y,w,h)  in centers1]
        centers1=[(x + (x+w))/2  for (x,y,w,h)  in centers1]

        for c in cnts2:
            centers2.append(cv2.boundingRect(c))


        center_point2=[((x + (x+w))/2,(y + (y+h))/2)  for (x,y,w,h)  in centers2]
        centers2 =[(x + (x+w))/2  for (x,y,w,h)  in centers2]

        #avarage the center
        centers1=sum(centers1)/len(centers1)
        centers2=sum(centers2)/len(centers2)

        # avarage the center
        center_point1 = (int(sum(elem[0] for elem in center_point1) / len(center_point1)),int(sum(elem[1] for elem in center_point1) / len(center_point1)))
        center_point2 = (int(sum(elem[0] for elem in center_point2) / len(center_point2)),int(sum(elem[1] for elem in center_point2) / len(center_point2)))

        #print(center_point1,center_point2)

        return (area1<area2,centers1>centers2),(center_point1,center_point2)

    # =========================FACE DETECION=======================================
    def denoise_img(self, image_list):

        print("denoising")

        if len(image_list) == 1:
            denoised = cv2.fastNlMeansDenoisingColored(image_list[0], None, 10, 10, 7, 21)

        else:

            # make the list odd
            if (len(image_list)) % 2 == 0: image_list.pop()

            middle = int(float(len(image_list)) / 2 - 0.5)

            # getting smallest images size
            width = 99999
            heigth = 99999

            for img in image_list:
                size = tuple(img.shape[1::-1])
                if size[0] < width: width = size[0]
                if size[1] < heigth: heigth = size[1]

            # resizing all images to the smallest one
            image_list = [cv2.resize(elem, (width, heigth)) for elem in image_list]

            imgToDenoiseIndex = middle
            temporalWindowSize = len(image_list)
            hColor = 3
            # print(temporalWindowSize, imgToDenoiseIndex)

            denoised = cv2.fastNlMeansDenoisingColoredMulti(image_list, imgToDenoiseIndex, temporalWindowSize,
                                                            hColor=hColor)
        print("denosed")


        return denoised

    def detect_face(self, img):

        scale_factor=1.4
        min_neight=3
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.frontal_face_cascade.detectMultiScale(img,scaleFactor=scale_factor,minNeighbors=min_neight)
        if len(faces) > 0:
            # print("face detcted!")
            return faces
        else:
            faces=self.profile_face_cascade.detectMultiScale(img,scaleFactor=scale_factor,minNeighbors=min_neight)
            if len(faces)> 0: return faces

        return ()

    def face_from_video(self, frames):
        """This funcion add a rectangle on recognized faces"""

        print("Starting face detector...")

        crop_frames = []
        faces = 0

        # for every frame in the video
        for frame in frames:

            # detect if there is a face
            face = self.detect_face(frame)

            # if there is a face
            if len(face) > 0:
                faces += 1
                # get the corners of the faces
                for (x, y, w, h) in face:

                    # if user want the face video too crop the image where face is detected
                    if self.face_photo_flag:
                        crop_frames.append(frame[y:y + h, x:x + w])





        if len(crop_frames)>0:

            self.face_recognizer.add_image_write(crop_frames)
            face=self.denoise_img(crop_frames)

        else: face=()

        print(str(faces) + " frames with faces detected")
        print("... face detector end")


        return face


    # =========================TELEGRAM BOT=======================================

    def motion_notifier(self, score, degub=False):
        """Function to notify user dor a detected movement"""
        to_send = "Movement detected!\n"
        if degub:
            to_send += "Score is " + str(score) + "\n"

        if self.video_flag and not self.face_photo_flag:
            to_send += "<b>Video</b> is <b>ON</b>...it may take a minute or two"
        elif self.face_photo_flag and self.video_flag:
            to_send += "Both <b>Video</b> and <b>Face Photo</b> are <b>ON</b> ... it may take a while"

        self.telegram_handler.send_message(to_send, parse_mode="HTML")

    def send_ground(self, specific_id, msg):
        self.telegram_handler.send_image(self.ground_frame,specific_id=specific_id, msg=msg)

    # =========================DEPRECATED=======================================


    def detect_motion_photo(self):
        initial_frame = self.frame[-1]
        sleep(self.delay)
        end_frame = self.frame[-1]

        # calculate diversity
        score = self.are_different(initial_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames
        if self.motion_flag and score:

            # send message
            self.motion_notifier(score)

            # take a new (more recent) frame
            prov = self.frame[-1]

            # take the time
            start = datetime.datetime.now()
            end = datetime.datetime.now()

            foud_face = False
            self.shotter.capture(True)

            print("INITIAL SCORE : " + str(score))

            # while the current frame and the initial one are different (aka some movement detected)
            while (self.are_different(initial_frame, prov)):

                print("in while")
                # check for the presence of a face in the frame
                if self.detect_face(prov):
                    # if face is detected send photo and exit while
                    self.send_image(prov, msg="Face detected!")
                    found_face = True
                    break

                # take another frame
                prov = self.frame[-1]

                # if time is exceeded exit while
                if (end - start).seconds > self.max_seconds_retries:
                    print("max seconds exceeded")
                    break

                # update current time in while loop
                end = datetime.datetime.now()

            if not foud_face:
                self.send_image(end_frame, msg="Face not detected")
            sleep(3)


class Telegram_handler(Thread):
    """Class to handle image/message/cideo sending throught telegram bot"""

    def __init__(self, bot):
        # init the thread
        Thread.__init__(self)

        self.bot = bot
        self.ids = self.get_ids(24978334)

        logger.info("Telegram handler started")

    def get_ids(self, fallback_id):
        """Get all the ids from the file"""
        # get ids form file
        print("getting ids from file")
        ids_path = "Resources/ids"

        if "ids" in os.listdir("Resources/"):
            with open(ids_path, "r+") as file:
                lines = file.readlines()

            # every line has the id as the first element of a split(,)
            ids = []
            for id in lines:
                ids.append(int(id.split(",")[0]))
            return ids


        else:
            return [fallback_id]

    def send_image(self, img, specific_id=0, msg=""):
        """Send an image to the ids """

        image_name = "image_to_send.png"

        ret = cv2.imwrite(image_name, img)

        if not ret and specific_id:
            self.send_message("There has been an error while writing the image",specific_id=specific_id)
            return
        elif not ret:
            if not ret and specific_id:
                self.send_message("There has been an error while writing the image")
                return

        else:
            with open(image_name, "rb") as file:
                if not specific_id:
                    for id in self.ids:
                        if msg:
                            self.bot.sendPhoto(id, file, caption=msg)
                        else:
                            self.bot.sendPhoto(id, file)
                else:
                    if msg:
                        self.bot.sendPhoto(specific_id, file, caption=msg)
                    else:
                        self.bot.sendPhoto(specific_id, file)

        os.remove(image_name)
        logger.info("Image sent")

    def send_message(self, msg,specific_id=0, parse_mode=""):

        if not specific_id:
            for id in self.ids:
                self.bot.sendMessage(id, msg, parse_mode=parse_mode)
        else:
            self.bot.sendMessage(specific_id, msg, parse_mode=parse_mode)

    def send_video(self, video_name, specific_id=0, msg=""):


        try:
            with open(video_name, "rb") as file:
                if not specific_id:
                    for id in self.ids:
                        if msg: self.bot.sendVideo(id, file, caption=msg)
                        else:self.bot.sendVideo(id, file)
                else:
                    if msg:
                        self.bot.sendVideo(specific_id, file, caption=msg)
                    else:
                        self.bot.sendVideo(specific_id, file)

            os.remove(video_name)

        except FileNotFoundError:
            self.send_message("The video could not be found ",specific_id=specific_id)

        logger.info("Video sent")

    def send_file(self,file_name,specific_id=0, msg=""):

        if (file_name in os.listdir(".")):
            with open(file_name, "rb") as file:
                if not specific_id:
                    for id in self.ids:
                        if msg: self.bot.sendDocument(id, file,caption=msg)
                        else:  self.bot.sendDocument(id, file)
                else:
                    if msg:
                        self.bot.sendDocument(specific_id, file, caption=msg)
                    else:
                        self.bot.sendDocument(specific_id, file)
        else:
            self.send_message("No log file detected!",specific_id=specific_id)

