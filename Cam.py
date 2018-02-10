import threading
from threading import Thread

import os
from skimage.measure import compare_ssim,compare_mse,compare_nrmse,compare_psnr
import cv2
from time import sleep
from datetime import datetime


class Cam_class:

    def __init__(self, bot):
        self.MAX_RETRIES = 4
        self.frames = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.shotter = Cam_shotter(self.frames)
        self.shotter.start()

        self.motion = Cam_movement(self.shotter, bot)
        self.motion.start()

    def capture_image(self, image_name):
        print("taking image")
        img = self.frames[-1]

        if isinstance(img, int):
            print("empty queue")
            return False
        # try to save the image
        ret = cv2.imwrite(image_name, img)

        # if the image was not saved return false
        if not ret: return False

        print("Image taken")
        return True

    def capture_video(self, video_name, seconds):
        # set camera resolution, fps and codec
        frame_width = 640
        frame_height = 480
        fps = 20
        print("initializing writer")

        out = cv2.VideoWriter(video_name, 0x00000021, fps, (frame_width, frame_height))

        print("writer initialized")

        #start capturing frames
        self.shotter.capture(True)
        #sleep
        sleep(seconds)
        #get catured frames
        to_write=self.shotter.capture(False)
        #write frame to file and release
        for elem in to_write:
            out.write(elem)
        out.release()




class Cam_shotter(Thread):
    """Class to take frames from camera"""

    def __init__(self, queue):

        # init the thread
        Thread.__init__(self)

        # get camera and queue
        self.cam_idx=0
        self.CAM = cv2.VideoCapture(self.cam_idx)
        self.queue = queue
        self.capture_bool = False
        self.capture_queue = []
        self.lock=threading.Lock()

    def run(self):
        """Main thread loop"""

        while True:

            # read frame form camera
            ret, img = self.CAM.read()

            # if frame has been read correctly add it to the end of the list
            if ret:
                # pop first element
                self.queue.pop(0)
                # append image at last
                self.queue.append(img)
                if self.capture_bool:
                    self.capture_queue.append(img)
                # print("saved")
            else:
                # try to reopen the camera
                print("not saved")
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
        print("reopening cam")
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
        print("close cam")
        self.CAM.release()

    def check_open_cam(self):
        """Function to open the camera stream"""
        print("checking cam")
        if not self.CAM.isOpened():
            print("cam was closed")
            self.CAM.open(0)
        else:
            print("cam was open")


class Cam_movement(Thread):
    """Class to detect movement from camera frames"""

    def __init__(self, shotter, bot):
        # init the thread
        Thread.__init__(self)

        self.shotter = shotter
        self.frame = shotter.queue
        self.bot = bot
        self.send_id = 24978334

        self.delay = 0.1
        self.diff_threshold = 2000
        self.notification = True
        self.image_name = "different.png"

        self.queue = []
        self.queue_len = 20

        self.face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml')
        self.max_seconds_retries = 5

        self.video_name = "detect_motion_video.mp4"

        self.resolution = (640, 480)  # width,height
        self.fps = 30
        self.out = cv2.VideoWriter(self.video_name, 0x00000021, self.fps, self.resolution)

    def run(self):

        while True:
            self.detect_motion_video()

    def detect_motion_photo(self):
        # get initial frame and and frame after delay seconds
        initial_frame = self.frame[-1]
        sleep(self.delay)
        end_frame = self.frame[-1]

        # if the notification is enable and there is a difference between the two frames
        if self.notification and self.are_different(initial_frame, end_frame):

            # take a new (more recent) frame
            prov = self.frame[-1]
            found_face = False

            # take the time
            start = datetime.now()
            end = datetime.now()

            # while the current frame and the initial one are different (aka some movement detected)
            while (self.are_different(initial_frame, prov)):

                print("in while")
                # check for the presence of a face in the frame
                if self.detect_face(prov):
                    # if face is detected send photo and exit while
                    self.send_image(prov, "Face detected!")
                    found_face = True
                    break

                # take another frame
                prov = self.frame[-1]

                # if time is exceeded exit while
                if (end - start).seconds > self.max_seconds_retries:
                    print("max seconds exceeded")
                    break

                # update current time in while loop
                end = datetime.now()

            if not found_face:
                self.send_image(end_frame, "Face not detected")
            sleep(3)

    def detect_motion_video(self):
        # get initial frame and and frame after delay seconds
        initial_frame = self.frame[-1]
        sleep(self.delay)
        end_frame = self.frame[-1]

        score=self.are_different(initial_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames
        if self.notification and score:

            #send message
            self.bot.sendMessage(self.send_id, "Movement detected with score : "+str(score))

            # take a new (more recent) frame
            prov = self.frame[-1]

            # take the time
            start = datetime.now()
            end = datetime.now()

            # create the file
            self.out.open(self.video_name, 0x00000021, self.fps, self.resolution)
            self.shotter.capture(True)

            # while the current frame and the initial one are different (aka some movement detected)
            while (self.are_different(initial_frame, prov)):

                print("in while")
                # check for the presence of a face in the frame
                # if self.detect_face(prov):
                #     found_face = True

                # take another frame
                prov = self.frame[-1]

                # if time is exceeded exit while
                if (end - start).seconds > self.max_seconds_retries:
                    print("max seconds exceeded")
                    break

                # update current time in while loop
                end = datetime.now()

                # sleep(1/self.fps)

            to_write = self.shotter.capture(False)
            for elem in to_write:
                self.out.write(elem)

            self.out.release()
            self.send_video(self.video_name)

            sleep(3)

    def detect_motion_old(self):

        initial_frame = self.frame[-1]
        sleep(self.delay)
        end_frame = self.frame[-1]

        if self.are_different(initial_frame, end_frame) and self.notification:
            self.bot.sendMessage(self.send_id, "Movement detected")
            self.send_image(end_frame)
            sleep(5)

    def are_different(self, img1, img2):

        if isinstance(img1, int) or isinstance(img2, int): return False
        similarity=self.get_similarity(img1,img2)
        if similarity>self.diff_threshold:
            return similarity

        else: return False


    def get_similarity(self, img1, img2):
        #start = datetime.now()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        #print("Convert to gray : " + str((datetime.now() - start).microseconds) + " microseconds")
        start = datetime.now()
        #(score, diff) = compare_ssim(img1, img2, full=True)
        score=compare_mse(img1,img2)
        print("COMPAIRISON : " + str((datetime.now() - start).microseconds) + " microseconds")

        print(score)

        return score

    def send_image(self, img, msg=""):

        ret = cv2.imwrite(self.image_name, img)
        if not ret:
            self.bot.sendMessage(self.send_id, "There has been an error while writing the image")
            return

        if msg:
            self.bot.sendMessage(self.send_id, msg)
        with open(self.image_name, "rb") as file:
            self.bot.sendPhoto(self.send_id, file)
        os.remove(self.image_name)

    def send_video(self, video_name, msg=""):

        with open(video_name, "rb") as file:
            if msg: self.bot.sendMessage(self.send_id, msg)
            self.bot.sendVideo(self.send_id, file)
        os.remove(video_name)

    def detect_face(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img)
        if len(faces) > 0:
            print("face detcted!")
            return True

        return False
