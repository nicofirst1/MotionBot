# import the necessary packages
from __future__ import print_function

import copy
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
    def __init__(self, vs, outputPath, darknet, face_reco):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath

        self.thread = None
        self.stopEvent = None
        self.darknet = darknet
        self.face_reco = face_reco

        # initialize the root window and image panel
        self.root = tki.Tk()

        self.panels = {
            'original': None,
            'darknet': None,
            'person': None,
            'face': None,
            'reco': None
        }

        # create a checkbutton, that when pressed, will take the current

        bott_panels = tki.Label(compound=tki.CENTER)
        bott_panels.pack(side="bottom", fill="both", padx=10, pady=10, expand="yes")

        self.face_reco_flag = tki.IntVar()

        check = tki.Checkbutton(bott_panels, text="save faces", variable=self.face_reco_flag)
        check.pack(side="right", fill="both",
                   expand="yes", padx=10,
                   pady=10)

        btn = tki.Button(bott_panels, text="Train",
                         command=self.face_reco.train_model)
        btn.pack(side="right", fill="both", expand="yes", padx=10,
                 pady=10)
        btn = tki.Button(bott_panels, text="Clean Faces",
                         command=self.face_reco.filter_all_images)
        btn.pack(side="right", fill="both", expand="yes", padx=10,
                 pady=10)
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def process_frame(self, original):

        frames = {
            'original': original,
            'darknet': None,
            'person': None,
            'face': None,
            'reco': None
        }

        segmentation = self.darknet.detect_img(original.copy())
        segmentation = [segmentation]

        frames['darknet'] = self.darknet.draw_bounds_list(copy.deepcopy(segmentation))[0]
        person = self.darknet.extract_faces(segmentation)

        if len(person):
            frames['person'] = person[0]

        frames['face'] = self.face_reco.find_faces(original.copy(), save=self.face_reco_flag.get())

        if frames['face'] is not None:
            reco = copy.deepcopy(original)
            prediction = self.face_reco.predict(reco)
            self.face_reco.show_prediction_labels_on_image(reco, prediction)
            frames['reco'] = reco

        return frames

    def convert_image(self, frames_dict):

        def convert_single(image):

            image = imutils.resize(image, width=300)

            # OpenCV represents images in BGR order; however PIL
            # represents images in RGB order, so we need to swap
            # the channels, then convert to PIL and ImageTk format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)

            return image

        for key, val in frames_dict.items():
            if val is not None:
                try:
                    frames_dict[key] = convert_single(val)
                except:
                    frames_dict[key] = None
        return frames_dict

    def update_panels(self, frames):

        for key in frames.keys():

            if not self.panels[key]:
                self.panels[key] = tki.Label(image=frames[key], text=f"{key}" + "\n" * 30, compound=tki.CENTER)
                self.panels[key].image = frames[key]
                self.panels[key].pack(side="left", padx=10, pady=10, )

            else:
                self.panels[key].configure(image=frames[key])
                self.panels[key].image = frames[key]

        time.sleep(0.01)

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
                original = self.vs.read()

                frames = self.process_frame(original)
                frames = self.convert_image(frames)
                self.update_panels(frames)



        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")

        self.stopEvent.set()
        self.vs.stop()

        for key in self.panels.keys():

            self.panels[key].configure(image=None)
            self.panels[key].image = None

        self.root.quit()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments

    # initialize the video stream and allow the camera sensor to warmup

    # start the app
    updater = Updater("545431258:AAHEocYDtLOQdZDCww6tQFSfq3p-xmWeyE8")
    disp = updater.dispatcher

    darknet = Darknet(True)
    darknet.start()

    face_reco = FaceRecognizer(disp)

    print("[INFO] warming up camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    pba = PhotoBoothApp(vs, ".", darknet, face_reco)
    pba.root.mainloop()
