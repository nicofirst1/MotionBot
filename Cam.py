import cv2
from time import sleep
from datetime import datetime

class Cam_class:

    def __init__(self):
        self.MAX_RETRIES=4
        self.CAM=cv2.VideoCapture(0)


    def reopen_cam(self):
        self.CAM.release()
        sleep(1.5)
        self.CAM=cv2.VideoCapture(0)
        self.check_open_cam()

    def close_cam(self):
        self.CAM.release()

    def check_open_cam(self):
        if not self.CAM.isOpened():
            self.CAM.open()

    def capture_image(self, image_name):
        max_ret = self.MAX_RETRIES
        print("taking image")

        #check if camera stream is opened
        if not self.CAM.isOpened():
            self.CAM.open(0)
        # try to read the image
        ret, img = self.CAM.read()

        # while the reading is unsuccesfull
        while not ret:
            print(max_ret)
            # read again and sleep
            sleep(0.5)
            ret, img = self.CAM.read()
            max_ret -= 1
            if not ret:
                self.reopen_cam()
            # if max retries is exceeded exit and release the stream
            if max_ret == 0:
                self.close_cam()
                return False

        # try to save the image
        ret = cv2.imwrite(image_name, img)
        max_ret = self.MAX_RETRIES

        while not ret:
            ret = cv2.imwrite(image_name, img)
            sleep(1)
            max_ret -= 1
            # if max retries is exceeded exit and release the stream

            if max_ret == 0:
                self.close_cam()
                return False

        self.close_cam()
        # sleep(2)
        print("Image taken")
        return True

    def capture_video(self, video_name, seconds):
        frame_width = 640
        frame_height = 480
        print(frame_height, frame_width)
        fps = 10
        out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                              (frame_width, frame_height))

        start = datetime.now()
        end = datetime.now()

        self.check_open_cam()

        while (True):
            ret, frame = self.CAM.read()

            if ret == True:

                # Write the frame into the file 'output.avi'
                out.write(frame)

            # Break the loop
            else:
                self.reopen_cam()
                pass

            if (end - start).seconds >= seconds:
                break

            end = datetime.now()

            # When everything done, release the video capture and video write objects
        self.close_cam()
        out.release()

