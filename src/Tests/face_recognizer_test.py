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
from Classes.Face_recognizer import FaceRecognizer, rename_images_index
from Path import Path as pt
from Utils.utils import read_token_psw


class PhotoBoothApp:
    def __init__(self, vs, outputPath, darknet, face_reco):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.queue = []

        rename_images_index(pt.UNK_DIR)

        self.thread = None
        self.stopEvent = None
        self.darknet = darknet
        self.face_reco = face_reco

        # initialize the root window and image panel
        self.root = tki.Tk()

        self.panels = {
            'original': None,
            'darknet': None,
            'reco': None
        }

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

        # create a left bttom panel
        check_panel = tki.Label(compound=tki.CENTER, master=bott_panels)
        check_panel.pack(side="right", fill="both", padx=10, pady=10, expand="yes")
        # check box
        self.face_save_flag = tki.BooleanVar()

        check = tki.Checkbutton(check_panel, text="save faces", variable=self.face_save_flag)
        check.pack(side="bottom", fill="both",
                   expand="yes", padx=10,
                   pady=10)

        self.darknet_flag = tki.BooleanVar()
        self.darknet_flag.set(False)

        check = tki.Checkbutton(check_panel, text="darknet", variable=self.darknet_flag)
        check.pack(side="bottom", fill="both",
                   expand="yes", padx=10,
                   pady=10)

        self.face_reco_flag = tki.BooleanVar()

        check = tki.Checkbutton(check_panel, text="face reco", variable=self.face_reco_flag)
        check.pack(side="bottom", fill="both",
                   expand="yes", padx=10,
                   pady=10)

        # train button
        btn = tki.Button(bott_panels, text="Train",
                         command=self.face_reco.train_model)
        btn.pack(side="right", fill="both", expand="yes", padx=10,
                 pady=10)

        # clean faces button
        btn = tki.Button(bott_panels, text="Clean Faces",
                         command=self.face_reco.filter_all_images)
        btn.pack(side="right", fill="both", expand="yes", padx=10,
                 pady=10)

    def process_frame(self, original):

        frames = {
            'original': original,
            'darknet': None,
            'reco': None
        }

        if self.darknet_flag.get():
            segmentation = self.darknet.detect_img(original.copy())
            segmentation = [segmentation]

            frames['darknet'] = self.darknet.draw_bounds_list(copy.deepcopy(segmentation))[0]

        if self.face_reco_flag.get():

            face = self.face_reco.find_faces(original.copy(), model="hog")

            if face is not None:

                self.queue.append(face)
                if len(self.queue) > 10 and self.face_save_flag.get():
                    self.face_reco.add_image_write(self.queue)
                    self.queue = []

                reco = copy.deepcopy(original)
                try:
                    prediction = self.face_reco.predict(reco)
                    self.face_reco.show_prediction_labels_on_image(reco, prediction)
                    frames['reco'] = reco
                except FileNotFoundError:
                    pass

        return frames

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
                self.panels[key] = tki.Label(image=frames[key], text=f"{key}" + "\n" * 30, compound=tki.CENTER)
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
                original = self.vs.read()

                frames = self.process_frame(original)
                frames = self.convert_image(frames)
                self.update_panels(frames)



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
        self.vs.stop()

        self.root.quit()


if __name__ == '__main__':
    # construct the argument parse and parse the arguments

    # initialize the video stream and allow the camera sensor to warmup

    # start the app
    token,psw=read_token_psw()
    updater = Updater(token)
    disp = updater.dispatcher

    face_reco = FaceRecognizer(disp)

    print("[INFO] warming up camera...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    darknet = Darknet(True)
    darknet.start()

    pba = PhotoBoothApp(vs, ".", darknet, face_reco)
    pba.root.mainloop()
