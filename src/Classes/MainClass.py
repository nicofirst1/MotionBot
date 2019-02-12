import logging
import os
from time import sleep
import cv2

from Classes.Darknet import Darknet
from src.Classes.Face_recognizer import FaceRecognizer
from src.Classes.CamMovement import CamMovement
from src.Classes.CamShotter import CamShotter
from src.Classes.TelegramHandler import TelegramHandler

# from memory_profiler import profile

logger = logging.getLogger('main_class')


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

        use_coco=False
        self.darknet=Darknet(use_coco)
        self.darknet.start()

        self.motion = CamMovement(self.shotter, self.telegram_handler, self.face_recognizer,self.darknet)
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

    def capture_video(self, video_name, seconds, user_id):
        """Get a video from the camera"""
        # set camera resolution, fps and codec
        frame_width = 640
        frame_height = 480
        fps = 20
        # print("initializing writer")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))

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

        self.telegram_handler.send_video(video_name, user_id, str(seconds) + " seconds record")

    def predict_face(self, img_path):
        """
            Predic the face in a photo and draw on it
        :param img_path: the path of the image to be predicted
        :return: string
        """

        # read the image and remove it from disk
        img = cv2.imread(img_path)
        os.remove(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find faces
        faces = self.motion.detect_face(gray, scale_factor=1.1, min_neight=5)

        print(faces)

        if faces is None:
            return faces

        # for every face predict the person and confidence
        for (x, y, w, h) in faces:
            name, confidence = self.face_recognizer.predict(img[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if confidence <= self.face_recognizer.auto_train_dist:
                cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "For sure!", (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            elif confidence <= self.face_recognizer.distance_thres:
                cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "Maybe...", (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

            else:
                cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "Just guessing", (x, y + h + 15), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        return img
