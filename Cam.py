import threading
import traceback
from threading import Thread
import os
import cv2
from time import sleep
import datetime
import logging

import sys
#from memory_profiler import profile
import numpy as np

from Face_recognizer import FaceRecognizer
from utils import time_profiler

logger = logging.getLogger('motionlog')


class MainClass:
    """This class is the one handling the thread initialization and the coordination between them.
    It also handles the telegram command to event execution

    Attributes:
        updater: the bot updater
        disp: the bot dispatcher
        bot : the telegram bot

        telegram_handler: the class that is in control of notifying the users about changes

        frames : a list of frames from which get the camera frames
        shotter : the class that takes frames from the camera

        face_recognizer : the class that handles the face recognition part

        motion: the class that handles the moving detection part


        """

    def __init__(self, updater):

        self.updater = updater
        self.disp = updater.dispatcher
        self.bot = self.disp.bot

        self.telegram_handler = TelegramHandler(self.bot)
        self.telegram_handler.start()

        self.frames = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.shotter = CamShotter(self.frames)
        self.shotter.start()

        self.face_recognizer = FaceRecognizer(self.disp)
        self.face_recognizer.start()

        self.motion = CamMovement(self.shotter, self.telegram_handler, self.face_recognizer)
        self.motion.start()
        logger.debug("Cam_class started")

    def capture_image(self, image_name):
        """Capture a frame from the camera
        """
        # print("taking image")
        img = self.frames[-2]

        if isinstance(img, int):
            print("empty queue")
            return False
        # try to save the image
        return cv2.imwrite(image_name, img)

    def stop(self):
        """Stop the execution of the threads and exit"""
        self.motion.stop()
        self.motion.join()

        self.face_recognizer.stop()
        self.face_recognizer.join()

        self.shotter.stop()
        self.shotter.join()

        logger.info("Stopping Cam class")
        return

    def capture_video(self, video_name, seconds,user_id):
        """Get a video from the camera"""
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

        self.telegram_handler.send_video(video_name, user_id,str(seconds) + " seconds record")


class CamShotter(Thread):
    """Class to take frames from camera, it is the only one who has access to the VideoCapture ojbect

    Attributes:
        stop_event : the event used to stop the class
        cam_idx : the camera index, usually 0
        CAM : the videoCapture object to take frames from the camera
        queue : a list of frames shared between the classes
        capture_bool : a flag value to start/stop/capturing all frames from the camera
        capture_queue : the list that will be holding all the frames
        lock : a lock object to lock the capture_queue
        camera_connected : a flag to notify the others thread that the camera is connected and they can start taking
        frames from the queue

    """

    def __init__(self, queue):

        # init the thread
        Thread.__init__(self)

        self.stop_event = threading.Event()

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

            # if the thread has been stopped
            if self.stopped():
                # release the cam object
                self.CAM.release()
                sleep(1)
                # delete the queue
                del self.queue[:]
                # log and return
                logger.info("Stopping Cam shotter")
                return

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

                #grayscale the future last frame
                try:
                    gray=self.queue[1]
                    self.queue[1]=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    #error_log = "Cv Error: " + str(e)
                    #print(error_log)
                    pass


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
        """Start/stop the frame capturing"""

        try:
            # if you want to capture the video
            if capture:
                # acquire the lock, empty the list and set the flag to true
                self.lock.acquire()
                self.capture_queue = []
                self.capture_bool = True
            else:
                # otherwise set the flag to false and release the lock
                self.capture_bool = False
                self.lock.release()
                return self.capture_queue
        except:
            self.lock.release()

    def reopen_cam(self):
        """Function to reopen the camera"""
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

    def stop(self):
        """Stop the thread"""
        self.stop_event.set()

    def stopped(self):
        """Check if thread has been stopped"""
        return self.stop_event.is_set()


class CamMovement(Thread):
    """Class to detect movement from camera frames, recognize faces, movement direction and much more
    Attributes:

        shotter : the cam_shotter object
        frame: the queue used by the shotter class

        telegram_handler : the telegram handler class

        face_recognizer: the face recognizer class

        delay : used delay between movement detection
        min_area : the minimum area of changes detected to be considered an actual change in the image
        ground_frame : the image used as the background to be compared with the current frames
        blur : the mean shift of the blur for the preprocessing of the images

        frontal_face_cascade : the object that detects frontal faces
        profile_face_cascade : the object that detects frontal faces
        face_size : the minimum window size to look for faces, the bigger the faster the program gets. But for distant
            people small values are to be taken into account

        max_seconds_retries : if a movement is detected for longer than max_seconds_retries the program will check for a
        background change, do not increase this parameter to much since it will slow down tremendously the program execution

        video_name : the name of the video to be send throught telegram
        resolution : the resolution of the video, do not change or telegram will not recognize the video as a gif
        fps : the frame per second of the video


        motion_flag : flag used to check if the user want to recieve a notification (can be set by telegram)
        video_flag :  flag used to check if the user want to recieve a video of the movement (can be set by telegram)
        face_photo_flag : flag used to check if the user want to recieve a photo of the faces in the video (can be set by telegram)
        debug_flag : flag used to check if the user want to recieve the debug images (can be set by telegram , it slows down the program)
        face_reco_falg : flag used to check if the user want to recieve the predicted face with the photo (can be set by telegram)

        faces_cnts : list of contours for detected faces
        max_blurrines : the maximum threshold for blurriness detection, discard face images with blur>max_blurrines

    """

    def __init__(self, shotter, telegram, face_recognizer):
        # init the thread
        Thread.__init__(self)
        self.stop_event = threading.Event()

        self.shotter = shotter
        self.frame = shotter.queue

        self.telegram_handler = telegram

        self.face_recognizer = face_recognizer

        self.delay = 0.1
        self.min_area = 2000
        self.ground_frame = 0
        self.blur = (10, 10)

        self.frontal_face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/lbpcascades/lbpcascade_frontalface.xml')
        self.profile_face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/lbpcascades/lbpcascade_profileface.xml')
        self.face_size=50

        self.max_seconds_retries = 10

        self.video_name = "detect_motion_video.mp4"
        self.resolution = (640, 480)  # width,height
        self.fps = 30

        self.video_flag = True
        self.face_photo_flag = True
        self.motion_flag = True
        self.debug_flag = False
        self.face_reco_falg = True

        self.resetting_ground = False

        self.faces_cnts=[]
        self.max_blurrines=100

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
                # detect a movement
                self.detect_motion_video()
            except:
                # on any exception log it and continue
                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                logger.error(''.join('!! ' + line for line in lines))  # Log it or whatever here

            if self.stopped():
                # if the thread has been stopped log and exit
                logger.info("Stopping Cam movement")
                return

    def stop(self):
        """Set the stop flag to true"""
        self.stop_event.set()

    def stopped(self):
        """Check for the flag value"""
        return self.stop_event.is_set()

    @time_profiler()
    def detect_motion_video(self):
        """Principal function for this class
        It takes a new frame from the queue and check if it different from the ground image
        If it is capture the movement
        """

        # wait for resetting to be over
        while self.resetting_ground:
            continue

        # get end frame after delay seconds
        sleep(self.delay)
        end_frame = self.frame[0]

        # calculate diversity
        score = self.are_different(self.ground_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames and the ground is not resetting
        if self.motion_flag and score and not self.resetting_ground:

            logger.info("Movement detected")
            # notify user
            self.motion_notifier(score)

            # do not capture video nor photo, just notification
            if not self.video_flag:
                return

            # start saving the frames
            self.shotter.capture(True)

            # while the current frame and the initial one are different (aka some movement detected)
            self.loop_difference(score, self.ground_frame, self.max_seconds_retries)

            # save the taken frames
            to_write = self.shotter.capture(False)

            # if the user wants the face in the movement
            if self.face_photo_flag:
                # take the face
                face = self.face_from_video(to_write)
                # if there are no faces found
                if face is None:
                    self.telegram_handler.send_message(msg="Face not found")

                else:
                    for elem in face:
                        self.telegram_handler.send_image(elem[2], msg="Found "+elem[0]+" with conficence = "+str(elem[1]))


            # send the original video too
            if not self.resetting_ground:
                print("Sending video...")
                # draw on the frames  and send video
                self.draw_on_frames(to_write)
                self.telegram_handler.send_video(self.video_name)
                print("...video sent")

    # =========================Movement=======================================

    def check_bk_changes(self, initial_frame, seconds):
        """Given the initial_frame and the max seconds the loop can run with, the function return false if movement is
        detected, otherwise, when the time has exceeded, it resets the back ground and return true"""
        # taking time
        start = datetime.datetime.now()
        end = datetime.datetime.now()
        # setting initial variables
        score = 0

        # setting initial frame
        #gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray = cv2.blur(initial_frame, self.blur, 0)

        # While there is movement
        while not score:

            # take another frame
            prov = self.frame[0]

            #print(prov.shape)

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

    def loop_difference(self, initial_score, initial_frame, seconds, retry=False):
        """Loop until the current frame is the same as the ground image or time is exceeded, retry is used to
        be make this approach robust to tiny changes"""

        if retry: print("retriyng")
        # take the time
        start = datetime.datetime.now()
        end = datetime.datetime.now()
        # get the initial score
        score = initial_score
        print("Start of difference loop")

        # while there is some changes and the ground is not being resetted
        while score and not self.resetting_ground:

            # take the already grayscaled frame
            prov = self.frame[0]
            #print(prov.shape)

            # check if images are different
            score = self.are_different(initial_frame, prov)

            # if time is exceeded exit while
            if (end - start).seconds > seconds:
                print("max seconds exceeded...checking for background changes")
                if not retry:
                    self.check_bk_changes(prov, 3)
                print("End of difference loop")
                return

            # update current time in while loop
            end = datetime.datetime.now()
            sleep(0.05)

        # it may be that there is no apparent motion
        if not retry:
            # wait a little and retry
            sleep(1.5)
            self.loop_difference(1, initial_frame, 1, True)

        print("End of difference loop")

    def are_different(self, grd_truth, img2):
        """Return whenever the difference in area between the ground image and the frame is grather than the
        threshold min_area."""

        cnts = self.compute_img_difference(grd_truth, img2)

        return any(cv2.contourArea(elem) > self.min_area for elem in cnts)

    def compute_img_difference(self, grd_truth, img2):
        """Compute te difference between the ground image and the frame passed as img2
        The ground image is supposed to be already preprocessed"""

        # blur and convert to grayscale
        #gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(img2, self.blur, 0)

        # compute the absolute difference between the current frame and
        # first frame
        try:
            frameDelta = cv2.absdiff(grd_truth, gray)
        except cv2.error as e:
            # catch any error and log
            error_log = "Cv Error: " + str(e) + "\ngrd_thruth : " + str(grd_truth.shape) + ", gray : " + str(
                gray.shape) + "\n"
            logger.error(error_log)
            print(error_log)
            # return true to not loose any movement
            return True

        # get the thresholded image
        thresh_original = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh_original, None, iterations=5)
        # get the contours of the changes
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if the debug flag is true send all the images
        if self.debug_flag:
            self.telegram_handler.send_image(frameDelta)
            self.telegram_handler.send_image(thresh_original, msg="Threshold Original")
            self.telegram_handler.send_image(thresh, msg="Threshold Dilated")
            self.telegram_handler.send_image(img2)

        # return the contours
        return cnts

    # =========================UTILS=======================================
    def draw_on_frames(self, frames, areas=True, date=True):
        """Function to draw on frames"""

        face_color = (0, 0, 255)  # red
        motion_color = (0, 255, 0)  # green
        line_tickness = 2

        # create the file
        out = cv2.VideoWriter(self.video_name, 0x00000021, self.fps, self.resolution)
        out.open(self.video_name, 0x00000021, self.fps, self.resolution)

        prov_cnts = 0
        idx = 0
        face_idx=0
        to_write = "Unkown - Unkown"
        print("Total frames to save : "+str(len(frames)))
        print("Total frames contours : "+str(len(self.faces_cnts)))
        for frame in frames:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.face_photo_flag:

                # take the corresponding contours for the frame
                face = self.faces_cnts[face_idx]
                face_idx+=1

                # if there is a face
                if face is not None:
                    # get the corners of the faces
                    for (x, y, w, h) in face:
                        # draw a rectangle around the corners
                        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, line_tickness)

            # draw movement
            if areas:
                cnts = self.compute_img_difference(self.ground_frame, gray)

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
                if not idx % 2:
                    prov_cnts = cnts
                elif len(prov_cnts) > 0 and len(cnts) > 0:
                    movement, _ = self.movement_direction(prov_cnts, cnts)
                    to_write = ""
                    if movement[0]:
                        to_write += "Incoming - "

                    else:
                        to_write += "Outgoing - "

                    if movement[1]:
                        to_write += "Left"

                    else:
                        to_write += "Right"

                    # cv2.circle(frame, center_points[0], 1, (255, 255, 0), 10,2)
                    # cv2.circle(frame, center_points[1], 1, (255, 255, 255), 10,2)

                cv2.putText(frame, to_write,
                            (frame.shape[1] - 250, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255),
                            1)

                idx += 1

            # add a date to the frame
            if date:
                # write time
                correct_date = datetime.datetime.now() + datetime.timedelta(hours=1)

                cv2.putText(frame, correct_date.strftime("%A %d %B %Y %H:%M:%S"),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

            # write frames on file
            out.write(frame)

        #empty the face contours list
        self.faces_cnts=[]

        # free file
        out.release()

    def reset_ground(self, msg):
        """Reset the ground truth image"""

        print("Reset ground image ...")
        # set the flag
        self.resetting_ground = True

        # convert to gray and blur
        gray = cv2.cvtColor(self.frame[-1], cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, self.blur, 0)
        # set the frame and notify
        self.ground_frame = gray
        self.telegram_handler.send_image(self.ground_frame, msg=msg)

        self.resetting_ground = False
        print("Done")

    @staticmethod
    def movement_direction(cnts1, cnts2):
        """Function to get the movement direction from two frames
        returns a tuple where the first elem is the outgoing/incoming 0/1, the second is right/left 0/1"""

        # the incoming outgoing movement is estimated throught the sum of the areas
        # if sum(areas1)<sum(areas2) the object is outgoing

        area1 = sum(cv2.contourArea(c) for c in cnts1)
        area2 = sum(cv2.contourArea(c) for c in cnts2)

        # print(area1,area2)

        # the left/right is given by the position of the averaged center of each area
        # if center1>center2 the object is moving right to left (so left)
        # get the centers of each area on the x axis
        centers1 = []
        centers2 = []

        # get the area
        for c in cnts1:
            centers1.append(cv2.boundingRect(c))

        # calculate the center point
        center_point1 = [((x + (x + w)) / 2, (y + (y + h)) / 2) for (x, y, w, h) in centers1]
        # calculate the mean position on the x axis
        centers1 = [(x + (x + w)) / 2 for (x, y, w, h) in centers1]

        # get the area
        for c in cnts2:
            centers2.append(cv2.boundingRect(c))

        center_point2 = [((x + (x + w)) / 2, (y + (y + h)) / 2) for (x, y, w, h) in centers2]
        # calculate the mean position on the x axis
        centers2 = [(x + (x + w)) / 2 for (x, y, w, h) in centers2]

        # avarage the center
        centers1 = sum(centers1) / len(centers1)
        centers2 = sum(centers2) / len(centers2)

        # avarage the center
        center_point1 = (int(sum(elem[0] for elem in center_point1) / len(center_point1)),
                         int(sum(elem[1] for elem in center_point1) / len(center_point1)))
        center_point2 = (int(sum(elem[0] for elem in center_point2) / len(center_point2)),
                         int(sum(elem[1] for elem in center_point2) / len(center_point2)))

        # print(center_point1,center_point2)

        return (area1 < area2, centers1 > centers2), (center_point1, center_point2)

    # =========================FACE DETECION=======================================

    #@time_profiler()
    def face_from_video(self, frames):
        """Detect faces from list of frames"""

        print("Starting face detection...")

        crop_frames = []
        faces = 0

        # for every frame in the video
        for frame in frames:

            # detect if there is a face
            face = self.detect_face(frame)
            self.faces_cnts.append(face)

            # if there is a face
            if face is not None:
                faces += 1
                # get the corners of the faces
                # if user want the face video too crop the image where face is detected
                if self.face_photo_flag:
                    for (x, y, w, h) in face:
                        blur_var=cv2.Laplacian(frame[y:y + h, x:x + w], cv2.CV_64F).var()
                        #print(blur_var)
                        #if the blur index of the image is grather than the threshold
                        if blur_var>=self.max_blurrines:
                            crop_frames.append(frame[y:y + h, x:x + w])
                            #self.telegram_handler.send_image(frame[y:y + h, x:x + w],msg="Blurr ok : "+str(blur_var))

                        else:
                            #self.telegram_handler.send_image(frame[y:y + h, x:x + w],msg="Too blurry : "+str(blur_var))
                            pass

        print(str(faces) + " frames with faces detected")
        print("... face detector end")

        # if there are some images with faces only
        if len(crop_frames) > 0:

            # if the users want to recognize faces
            if self.face_reco_falg:
                # try to move te images to the Unknown folder
                if not self.face_recognizer.add_image_write(crop_frames):
                    print("Error during the insertion of face images into dir")
                    logger.error("Error during the insertion of face images into dir")

            # get the final face image denoising the others
            faces_img=self.face_recognizer.predict_multi(crop_frames)

        else:
            faces_img = []

        return faces_img

    def detect_face(self, img):
        """Detect faces using the cascades"""
        # setting the parameters
        scale_factor = 1.4
        min_neight = 3
        min_size=(self.face_size,self.face_size)
        # converting to gray
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # try to detect the front face
        faces = self.frontal_face_cascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neight,minSize=min_size)
        if len(faces) > 0:
            # print("face detcted!")
            return faces
        # else:
        #     # if there are no frontface, detect the profile ones
        #     faces = self.profile_face_cascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neight)
        #     if len(faces) > 0:
        #         return faces

        return None


    # =========================TELEGRAM BOT=======================================

    def motion_notifier(self, score, degub=False):
        """Function to notify user for a detected movement"""
        to_send = "Movement detected!\n"
        if degub:
            to_send += "Score is " + str(score) + "\n"

        if self.face_photo_flag and self.video_flag and self.face_reco_falg:
            to_send += "<b>Video</b>, <b>Face Photo</b> and <b>Face Reco</b> are <b>ON</b> ... it may take a while"
        elif self.face_photo_flag and self.video_flag:
            to_send += "Both <b>Video</b> and <b>Face Photo</b> are <b>ON</b> ... it may take a while"
        elif self.video_flag:
            to_send += "<b>Video</b> is <b>ON</b>...it may take a minute or two"

        self.telegram_handler.send_message(to_send, parse_mode="HTML")

    def send_ground(self, specific_id, msg):
        """Send the ground image to the users"""
        self.telegram_handler.send_image(self.ground_frame, specific_id=specific_id, msg=msg)

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
            while self.are_different(initial_frame, prov):

                print("in while")
                # check for the presence of a face in the frame
                if self.detect_face(prov):
                    # if face is detected send photo and exit while
                    self.telegram_handler.send_image(prov, msg="Face detected!")
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
                self.telegram_handler.send_image(end_frame, msg="Face not detected")
            sleep(3)


    @staticmethod
    def denoise_img(image_list):
        """Denoise one or multiple images"""

        print("denoising")

        if len(image_list) == 1:
            denoised = cv2.fastNlMeansDenoisingColored(image_list[0], None, 10, 10, 7, 21)

        else:

            # make the list odd
            if (len(image_list)) % 2 == 0:
                image_list.pop()
            # get the middle element
            middle = int(float(len(image_list)) / 2 - 0.5)

            width = sys.maxsize
            heigth = sys.maxsize

            # getting smallest images size
            for img in image_list:
                size = tuple(img.shape[1::-1])
                if size[0] < width: width = size[0]
                if size[1] < heigth: heigth = size[1]

            # resizing all images to the smallest one
            image_list = [cv2.resize(elem, (width, heigth)) for elem in image_list]

            imgToDenoiseIndex = middle
            temporalWindowSize = len(image_list)
            hColor = 3
            searchWindowSize=17
            hForColorComponents=1
            # print(temporalWindowSize, imgToDenoiseIndex)

            denoised = cv2.fastNlMeansDenoisingColoredMulti(image_list, imgToDenoiseIndex, temporalWindowSize,
                                                            hColor=hColor,searchWindowSize=searchWindowSize,
                                                            hForColorComponents=hForColorComponents)
        print("denosed")

        return denoised


class TelegramHandler(Thread):
    """Class to handle image/message/video sending throught telegram bot"""

    def __init__(self, bot):
        # init the thread
        Thread.__init__(self)

        self.bot = bot
        self.default_id = 24978334
        self.ids = self.get_ids(self.default_id)

        #print(self.ids)

        logger.info("Telegram handler started")

    @staticmethod
    def get_ids(fallback_id):
        """Get all the ids from the file"""
        # get ids form file
        print("getting ids from file")
        ids_path = "Resources/ids"

        # if there are some ids in the file get them
        if "ids" in os.listdir("Resources/"):
            with open(ids_path, "r+") as file:
                lines = file.readlines()

            # every line has the id as the first element of a split(,)
            ids = []
            for user_id in lines:
                if int(user_id.split(",")[1]):
                    ids.append(int(user_id.split(",")[0]))
            return ids

        else:
            # return the default id
            return [fallback_id]

    def send_image(self, img, specific_id=0, msg=""):
        """Send an image to the ids """

        image_name = "image_to_send.png"

        ret = cv2.imwrite(image_name, img)

        if not ret and specific_id:
            self.send_message("There has been an error while writing the image", specific_id=specific_id)
            return
        elif not ret:
            if not ret and specific_id:
                self.send_message("There has been an error while writing the image")
                return

        else:
            with open(image_name, "rb") as file:
                if not specific_id:
                    for user_id in self.ids:
                        if msg:
                            self.bot.sendPhoto(user_id, file, caption=msg)
                        else:
                            self.bot.sendPhoto(user_id, file)
                else:
                    if msg:
                        self.bot.sendPhoto(specific_id, file, caption=msg)
                    else:
                        self.bot.sendPhoto(specific_id, file)

        os.remove(image_name)
        logger.info("Image sent")

    def send_message(self, msg, specific_id=0, parse_mode=""):
        """Send a message to the ids"""

        if not specific_id:
            for user_id in self.ids:
                self.bot.sendMessage(user_id, msg, parse_mode=parse_mode)
        else:
            self.bot.sendMessage(specific_id, msg, parse_mode=parse_mode)

    def send_video(self, video_name, specific_id=0, msg=""):
        """Send a video to the ids"""

        try:
            with open(video_name, "rb") as file:
                if not specific_id:
                    for user_id in self.ids:
                        if msg:
                            self.bot.sendVideo(user_id, file, caption=msg)
                        else:
                            self.bot.sendVideo(user_id, file)
                else:
                    if msg:
                        self.bot.sendVideo(specific_id, file, caption=msg)
                    else:
                        self.bot.sendVideo(specific_id, file)

            os.remove(video_name)

        except FileNotFoundError:
            self.send_message("The video could not be found ", specific_id=specific_id)

        logger.info("Video sent")

    def send_file(self, file_name, specific_id=0, msg=""):
        """Send a file to the ids"""

        if file_name in os.listdir("."):
            with open(file_name, "rb") as file:
                if not specific_id:
                    for user_id in self.ids:
                        if msg:
                            self.bot.sendDocument(user_id, file, caption=msg)
                        else:
                            self.bot.sendDocument(user_id, file)
                else:
                    if msg:
                        self.bot.sendDocument(specific_id, file, caption=msg)
                    else:
                        self.bot.sendDocument(specific_id, file)
        else:
            self.send_message("No log file detected!", specific_id=specific_id)
