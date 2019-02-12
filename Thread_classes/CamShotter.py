import logging
import threading
from threading import Thread
from time import sleep

import cv2

# from memory_profiler import profile

logger = logging.getLogger('cam_shotter')


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
        self.cam_idx = 1
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

                # grayscale the future last frame
                try:
                    gray = self.queue[1]
                    self.queue[1] = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                except cv2.error as e:
                    error_log = "Cv Error: " + str(e)
                    logger.info(error_log)
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
