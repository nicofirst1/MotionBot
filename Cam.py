from threading import Thread

import os
from skimage.measure import compare_ssim
import cv2
from time import sleep
from datetime import datetime

class Cam_class:

    def __init__(self,bot):
        self.MAX_RETRIES=4
        self.frames = [0,0,0,0,0,0,0,0,0,0]


        self.shotter=Cam_shotter(self.frames)
        self.shotter.start()


        self.motion=Cam_movement(self.frames,bot)
        self.motion.start()







    def capture_image(self,image_name):
        print("taking image")
        img = self.frames[-1]

        if isinstance(img, int):
            print("empy queue")
            return False
        # try to save the image
        ret = cv2.imwrite(image_name, img)

        #if the image was not saved return false
        if not ret: return False

        print("Image taken")
        return True

    def capture_video(self, video_name, seconds):
        frame_width = 640
        frame_height = 480
        print(frame_height, frame_width)
        fps = 20
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MP4V'), fps,
                              (frame_width, frame_height))

        start = datetime.now()
        end = datetime.now()


        while (True):

            frame=self.frames[-1]
            out.write(frame)


            if (end - start).seconds >= seconds:
                break

            end = datetime.now()

            sleep(1/fps)

        # When everything done, release the video capture and video write objects
        out.release()


class Cam_shotter(Thread):
    """Class to take frames from camera"""
    def __init__(self, queue):

        #init the thread
        Thread.__init__(self)


        #get camera and queue
        self.CAM=cv2.VideoCapture(0)
        self.queue=queue

    def run(self):


        while True:

            ret, img = self.CAM.read()

            if ret:
                # pop first element
                self.queue.pop(0)
                #append image at last
                self.queue.append(img)
                #print("saved")
            else:
                print("not saved")
                self.reopen_cam()


            #sleep(0.01)

    def reopen_cam(self):
        print("reopening cam")
        self.CAM.release()
        sleep(2)
        self.CAM = cv2.VideoCapture(0)
        sleep(2)

        self.check_open_cam()
        self.h=self.CAM.get(4)
        self.w=self.CAM.get(3)

    def close_cam(self):
        print("close cam")
        self.CAM.release()

    def check_open_cam(self):
        print("checking cam")
        if not self.CAM.isOpened():
            print("cam was closed")
            self.CAM.open(0)
        else:
            print("cam was open")


class Cam_movement(Thread):
    """Class to detect movement from camera frames"""

    def __init__(self, frames, bot):
        # init the thread
        Thread.__init__(self)

        self.frame=frames
        self.bot=bot
        self.send_id=24978334

        self.delay=0.5
        self.diff_threshold=0.7
        self.notification=True
        self.image_name="different.png"

        self.queue=[]
        self.queue_len=20

        self.face_cascade = cv2.CascadeClassifier('/home/pi/InstallationPackages/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml')

    def run(self):


        while True:



            initial_frame=self.frame[-1]
            sleep(self.delay)
            end_frame=self.frame[-1]



            # if self.are_different(initial_frame,end_frame) and self.notification:
            #
            #     self.bot.sendMessage(self.send_id, "Ho rilevato un movimento!")
            #     self.send_image(end_frame)
            #     sleep(5)

            if self.are_different(initial_frame, end_frame) and self.notification:
                prov=self.frame[-1]
                found_face=False
                while (not self.are_different(initial_frame, prov)):
                    print("in while")
                    if self.detect_face(prov):
                        self.send_image(prov,"Faccia rilevata!")
                        found_face=True
                    prov=self.frame[-1]
                if not found_face:
                    self.send_image(end_frame,"Faccia non rilevata")
                sleep(3)


    def are_different(self, img1, img2):

        if isinstance(img1,int) or isinstance(img2,int): return False

        return self.get_similarity(img1,img2)<self.diff_threshold


    def get_similarity(self, img1,img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(img1, img2, full=True)
        print(score)

        return score

    def send_image(self, img, msg=""):

        ret = cv2.imwrite(self.image_name, img)
        if not ret:
            self.bot.sendMessage(self.send_id, "Errore durante la scrittura dell'immagine")
            return

        if msg:
            self.bot.sendMessage(self.send_id,msg)
        with open(self.image_name, "rb") as file:
            self.bot.sendPhoto(self.send_id, file)
        os.remove(self.image_name)



    def detect_face(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img)
        if faces.any():
            print("face detcted!")
            return True

        return False