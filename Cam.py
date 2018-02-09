from threading import Thread

import cv2
from time import sleep
from datetime import datetime

class Cam_class:

    def __init__(self):
        self.MAX_RETRIES=4
        self.frames = [0,0,0,0,0,0,0,0,0,0]


        self.thread=Cam_thread(self.frames)
        self.thread.start()





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
        frame_width = self.thread.w
        frame_height = self.thread.h
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


class Cam_thread(Thread):
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

