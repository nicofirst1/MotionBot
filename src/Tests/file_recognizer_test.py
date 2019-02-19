# import the necessary packages
from __future__ import print_function

import threading
import time
import tkinter as tki
from tkinter.filedialog import askopenfilename

import cv2
import face_recognition
import imutils
from PIL import Image
from PIL import ImageTk
# import the necessary packages
from telegram.ext import Updater
from tqdm import tqdm

from Classes.Face_recognizer import FaceRecognizer, filter_similar_images, filter_all_images, \
    show_prediction_labels_on_image
from Utils.utils import read_token_psw


def convert_image(frames_dict):
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
        self.face_save_flag = None

        # initialize the root window and image panel
        self.root = tki.Tk()

        self.panels = {
            'original': None,
            'reco': None
        }

        self.frames = None
        # prepare the gui
        self.pack_gui()

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)

    def pack_gui(self):

        # create a bottom panel
        bott_panels = tki.Label(compound=tki.CENTER)
        bott_panels.pack(side="bottom", fill="both", padx=10, pady=10, expand="yes")

        # create a left bttom panel
        check_panel = tki.Label(compound=tki.CENTER, master=bott_panels)
        check_panel.pack(side="right", fill="both", padx=10, pady=10, expand="yes")

        # create a left bttom panel
        radio_panel = tki.Label(compound=tki.CENTER, master=bott_panels)
        radio_panel.pack(side="right", fill="both", padx=10, pady=10, expand="yes")

        def build_button():
            # train button
            btn = tki.Button(bott_panels, text="Train",
                             command=self.face_reco.train_model)
            btn.pack(side="right", fill="both", expand="yes", padx=10,
                     pady=10)

            # clean faces button
            btn = tki.Button(bott_panels, text="Clean Faces",
                             command=filter_all_images)
            btn.pack(side="right", fill="both", expand="yes", padx=10,
                     pady=10)

            # choose photo button
            btn = tki.Button(bott_panels, text="Choose File",
                             command=self.choose_file)
            btn.pack(side="left", fill="both", expand="yes", padx=10,
                     pady=10)

        def build_check():
            # check box

            face_save_flag = tki.BooleanVar()

            check = tki.Checkbutton(check_panel, text="save faces", variable=face_save_flag)
            check.pack(side="bottom", fill="both",
                       expand="yes", padx=10,
                       pady=10)

            return face_save_flag

        def build_radio():
            """
            Build radio choices
            :return:
            """
            v = tki.IntVar()
            v.set(2)

            algorithms = [
                ('svm', 0),
                ('knn', 1),
                ('top n', 2),
                ('min sum', 3)
            ]

            for val, language in enumerate(algorithms):
                tki.Radiobutton(radio_panel,
                                text=language,
                                padx=20,
                                variable=v,
                                command=self.set_algorithm,
                                value=val).pack(anchor=tki.W)

            return v

        build_button()

        self.face_save_flag = build_check()

        self.algorithm = build_radio()

    def set_algorithm(self):

        self.face_reco.switch_classifier(self.algorithm.get())

        if self.frames is not None:
            self.predict_photo(self.frames)

    def choose_file(self):
        """
        Prompt the choosing of a file
        :return:
        """

        # get the file
        filename = askopenfilename()

        def capture_video(path):
            """
            Return a list of frames from a video
            :param path: the path to the video
            :return:
            """

            print("Reading video...")
            cap = cv2.VideoCapture(path)
            frames = []
            ret = True
            while ret:
                ret, frame = cap.read()

                frames.append(frame)

            cap.release()

            return frames[:-1]

        try:
            # if the file is an image
            img = cv2.imread(filename)

        except:
            # if the file is not valid return
            return

        # if the img is none then it is a video
        if img is None:
            imgs = capture_video(filename)

            if self.face_save_flag.get():
                faces = []
                # find faces and save them
                print("Detecting faces...")
                for img in imgs:
                    face = self.face_reco.find_faces(img, save=False, model="cnn")
                    if face is not None:
                        faces.append(face)

                self.face_reco.add_image_write(faces)
                filter_all_images()

            img = imgs


        else:
            img = [img]

        print("Predicting faces...")
        update_frames=[]
        for frame in tqdm(img):
            uf=self.predict_photo(frame)
            update_frames.append(uf)

        self.frames=update_frames

    def predict_photo(self, img):
        """
        Execute recognition of an image
        :param img: the image
        :return:
        """
        # find faces
        prediction = self.face_reco.predict(img)
        pred_img = img.copy()
        # update frames
        frames = {
            'original': img,
            'reco': pred_img
        }

        if not len(prediction):
            return convert_image(frames)

        # write prediction on image
        show_prediction_labels_on_image(pred_img, prediction)


        # convert and update panel
        frames = convert_image(frames)

        self.update_panels(frames)

        return frames

    def update_panels(self, frames):

        for key in frames.keys():

            if not self.panels[key]:
                self.panels[key] = tki.Label(image=frames[key], compound=tki.CENTER)
                self.panels[key].image = frames[key]
                self.panels[key].pack(side="left", padx=10, pady=10, )

            else:
                self.panels[key].configure(image=frames[key])
                self.panels[key].image = frames[key]

            #time.sleep(0.01)

    def video_loop(self):

        # keep looping over frames until we are instructed to stop
        while not self.stopEvent.is_set():
            # grab the frame from the video stream and resize it to
            # have a maximum width of 300 pixels

            if self.frames is not None:
                for frame in self.frames:
                    self.update_panels(frame)
                    time.sleep(0.05)

                time.sleep(0.5)

    def on_close(self):
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
