# import the necessary packages
from __future__ import print_function

import threading
import time
import tkinter as tki
from tkinter.filedialog import askopenfilename

import cv2
import imutils
from PIL import Image
from PIL import ImageTk
# import the necessary packages
from telegram.ext import Updater

from Classes.Face_recognizer import FaceRecognizer
from Utils.utils import read_token_psw


class PhotoReco:
    def __init__(self, face_reco):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event

        self.algorithm = None
        self.queue = []

        self.thread = None
        self.stopEvent = None
        self.face_reco = face_reco

        # initialize the root window and image panel
        self.root = tki.Tk()

        self.panels = {
            'original': None,
            'reco': None
        }

        self.img = None
        # prepare the gui
        self.pack_gui()

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def pack_gui(self):

        # create a bottom panel
        bott_panels = tki.Label(compound=tki.CENTER)
        bott_panels.pack(side="bottom", fill="both", padx=10, pady=10, expand="yes")

        btn = tki.Button(bott_panels, text="Choose Photo",
                         command=self.choose_photo)
        btn.pack(side="left", fill="both", expand="yes", padx=10,
                 pady=10)

        v = tki.IntVar()
        v.set(2)
        self.algorithm = v

        algorithms = [
            ('svm', 0),
            ('knn', 1),
            ('top n', 2),
            ('min sum', 3)
        ]

        for val, language in enumerate(algorithms):
            tki.Radiobutton(bott_panels,
                            text=language,
                            padx=20,
                            variable=v,
                            command=self.set_algorithm,
                            value=val).pack(anchor=tki.W)

    def set_algorithm(self):

        self.face_reco.switch_classificator(self.algorithm.get())

        if self.img is not None:
            self.predict(self.img)

    def choose_photo(self):
        filename = askopenfilename()

        try:
            img = cv2.imread(filename)
        except Exception:
            return
        self.img = img
        self.predict(img)

    def predict(self, img):
        # find faces
        prediction = self.face_reco.predict(img)
        pred_img = img.copy()

        if not len(prediction):
            return None

        self.face_reco.show_prediction_labels_on_image(pred_img, prediction)

        frames = {
            'original': img,
            'reco': pred_img
        }

        frames = self.convert_image(frames)
        self.update_panels(frames)

    def convert_image(self, frames_dict):

        def convert_single(image):

            image = imutils.resize(image, width=500)

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
                self.panels[key] = tki.Label(image=frames[key], compound=tki.CENTER)
                self.panels[key].image = frames[key]
                self.panels[key].pack(side="left", padx=10, pady=10, )

            else:
                self.panels[key].configure(image=frames[key])
                self.panels[key].image = frames[key]

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
                time.sleep(0.01)




        except RuntimeError as e:
            print(f"[INFO] caught a RuntimeError: {e}")

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")

        for key in self.panels.keys():
            self.panels[key].configure(image=None)
            self.panels[key].image = None

        self.stopEvent.set()

        self.root.quit()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments

    # initialize the video stream and allow the camera sensor to warmup

    # start the app
    token, psw = read_token_psw()
    updater = Updater(token)
    disp = updater.dispatcher

    face_reco = FaceRecognizer(disp)

    pba = PhotoReco(face_reco)
    pba.root.mainloop()
