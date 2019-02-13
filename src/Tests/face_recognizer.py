# import the necessary packages
from __future__ import print_function

import datetime
import os
import threading
import time
import tkinter as tki

import cv2
import imutils
from PIL import Image
from PIL import ImageTk
# import the necessary packages
from imutils.video import VideoStream
from telegram.ext import Updater

from Classes.Darknet import Darknet
from Classes.Face_recognizer import FaceRecognizer


class PhotoBoothApp:
    def __init__(self, vs, outputPath,darknet,face_reco):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.face=None
        self.thread = None
        self.stopEvent = None
        self.darknet=darknet
        self.face_reco=face_reco


        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        self.panel_face = None
        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Snapshot!",
                         command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,
                 pady=10)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)


    def process_frame(self):

        segmentation= self.darknet.detect_img(self.frame)
        segmentation=[segmentation]
        self.frame=self.darknet.draw_bounds_list(segmentation)[0]
        self.face=self.darknet.extract_faces(segmentation)

        if not len(self.face):
            self.face=self.frame

        else:
            self.face=self.face[0]




    def convert_image(self, image):

        image = imutils.resize(image, width=300)

        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        return image



    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()

                self.process_frame()



                frame=self.convert_image(self.frame)
                try:
                    face=self.convert_image(self.face)
                except Exception:
                    face=frame

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(image=frame)
                    self.panel.image = frame
                    self.panel.pack(side="left", padx=10, pady=10)

                    self.panel_face = tki.Label(image=face)
                    self.panel_face.image = face
                    self.panel_face.pack(side="right", padx=10, pady=10)


                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=frame)
                    self.panel.image = frame

                    self.panel_face.configure(image=face)
                    self.panel_face.image = face

                    time.sleep(0.01)

        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))

        # save the file
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(filename))

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()



if __name__ == '__main__':
    # construct the argument parse and parse the arguments

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] warming up camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # start the app
    updater = Updater("545431258:AAHEocYDtLOQdZDCww6tQFSfq3p-xmWeyE8")
    disp = updater.dispatcher

    darknet = Darknet(True)
    darknet.start()

    face_reco = FaceRecognizer(disp)


    pba = PhotoBoothApp(vs, ".",darknet,face_reco)
    pba.root.mainloop()
