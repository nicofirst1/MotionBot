import threading
from threading import Thread

import os
from skimage.measure import compare_ssim, compare_mse, compare_nrmse, compare_psnr
import cv2
from time import sleep
from datetime import datetime


class Cam_class:

    def __init__(self, bot):
        self.MAX_RETRIES = 4
        self.frames = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.shotter = Cam_shotter(self.frames)
        self.shotter.start()

        self.motion = Cam_movement(self.shotter, bot)
        self.motion.start()

    def capture_image(self, image_name):
        # print("taking image")
        img = self.frames[-1]

        if isinstance(img, int):
            print("empty queue")
            return False
        # try to save the image
        ret = cv2.imwrite(image_name, img)

        # if the image was not saved return false
        if not ret: return False

        # print("Image taken")
        return True

    def capture_video(self, video_name, seconds):
        # set camera resolution, fps and codec
        frame_width = 640
        frame_height = 480
        fps = 20
        # print("initializing writer")

        out = cv2.VideoWriter(video_name, 0x00000021, fps, (frame_width, frame_height))

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


class Cam_shotter(Thread):
    """Class to take frames from camera"""

    def __init__(self, queue):

        # init the thread
        Thread.__init__(self)

        # get camera and queue
        self.cam_idx = 0
        self.CAM = cv2.VideoCapture(self.cam_idx)
        self.queue = queue
        self.capture_bool = False
        self.capture_queue = []
        self.lock = threading.Lock()

    def run(self):
        """Main thread loop"""

        first_ret = False

        while True:

            # read frame form camera
            ret, img = self.CAM.read()

            # if frame has been read correctly add it to the end of the list
            if ret:
                # if it is the first time that the class reads an image
                if not first_ret:
                    print("camera connected")
                    first_ret = True
                    # sleep to wait for auto-focus/brightness
                    sleep(3)
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

        try:
            if capture:
                self.lock.acquire()
                self.capture_queue = []
                self.capture_bool = True
            else:
                self.capture_bool = False
                self.lock.release()
                return self.capture_queue
        except:
            self.lock.release()

    def reopen_cam(self):
        """Function to reopen the camera"""
        # print("reopening cam")
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


class Cam_movement(Thread):
    """Class to detect movement from camera frames"""

    def __init__(self, shotter, bot):
        # init the thread
        Thread.__init__(self)

        self.shotter = shotter
        self.frame = shotter.queue
        self.bot = bot
        self.send_id = 24978334

        self.delay = 0.1
        self.diff_threshold = 27.0
        self.image_name = "different.png"

        self.queue = []
        self.queue_len = 20

        self.face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt_tree.xml')
        self.max_seconds_retries = 5

        self.video_name = "detect_motion_video.mp4"

        self.resolution = (640, 480)  # width,height
        self.fps = 30
        self.out = cv2.VideoWriter(self.video_name, 0x00000021, self.fps, self.resolution)

        self.faces_video_flag = True
        self.face_photo_flag = True
        self.motion_flag = True

    def run(self):

        while True:
            self.detect_motion_video()

    def detect_motion_photo(self):
        initial_frame = self.frame[-1]
        sleep(self.delay)
        end_frame = self.frame[-1]

        # calculate diversity
        score = self.are_different(initial_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames
        if self.motion_flag and score:

            # send message
            self.motion_notifier(score)

            # take a new (more recent) frame
            prov = self.frame[-1]

            # take the time
            start = datetime.now()
            end = datetime.now()

            foud_face=False
            self.shotter.capture(True)

            print("INITIAL SCORE : "+str(score))

            # while the current frame and the initial one are different (aka some movement detected)
            while (self.are_different(initial_frame, prov)):

                print("in while")
                # check for the presence of a face in the frame
                if self.detect_face(prov):
                    # if face is detected send photo and exit while
                    self.send_image(prov, "Face detected!")
                    found_face = True
                    break

                # take another frame
                prov = self.frame[-1]

                # if time is exceeded exit while
                if (end - start).seconds > self.max_seconds_retries:
                    print("max seconds exceeded")
                    break

                # update current time in while loop
                end = datetime.now()

            if not foud_face:
                self.send_image(end_frame, "Face not detected")
            sleep(3)

    def detect_motion_video(self):
        # get initial frame and and frame after delay seconds
        initial_frame = self.frame[-1]
        sleep(self.delay)
        end_frame = self.frame[-1]

        # calculate diversity
        score = self.are_different(initial_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames
        if self.motion_flag and score:

            # send message
            self.motion_notifier(score)

            # take a new (more recent) frame
            prov = self.frame[-1]

            # take the time
            start = datetime.now()
            end = datetime.now()

            # create the file
            if self.faces_video_flag:
                self.out.open(self.video_name, 0x00000021, self.fps, self.resolution)
            self.shotter.capture(True)

            self.send_image(initial_frame,"initial frame")
            self.send_image(end_frame,"end frame")
            # while the current frame and the initial one are different (aka some movement detected)
            while (score):

                score = self.are_different(initial_frame, prov,22)
                print(score)
                # take another frame
                prov = self.frame[-1]

                # if time is exceeded exit while
                if (end - start).seconds > self.max_seconds_retries:
                    print("max seconds exceeded")
                    break

                # update current time in while loop
                end = datetime.now()

            print("End of while loop")
            to_write = self.shotter.capture(False)

            if self.faces_video_flag or self.face_photo_flag:
                to_write, cropped_frames = self.face_on_video(to_write)

                # if the face video is avaiable
                if len(cropped_frames) > 0 and self.face_photo_flag:
                    # write it, release the stream
                    denoised=self.denoise_img(cropped_frames)
                    self.send_image(denoised, "Frames : " + str(len(cropped_frames)))
                elif len(cropped_frames)==0:
                    self.bot.sendMessage(self.send_id,"No faces found")

            # send the original video too
            if self.faces_video_flag:
                for elem in to_write:
                    self.out.write(elem)
                self.out.release()
                self.send_video(self.video_name)

            sleep(3)

    def motion_notifier(self, score, degub=True):

        to_send="Movement detected!\n"
        if degub:
            to_send+="Score is "+str(score)+"\n"




        if self.faces_video_flag and not self.face_photo_flag:
            to_send+="<b>Face Video</b> is <b>ON</b>...it may take a minute or two"
        elif self.face_photo_flag and self.faces_video_flag:
            to_send+="Both <b>Face Video</b> and <b>Face Photo</b> are <b>ON</b> ... it may take a while"


        self.bot.sendMessage(self.send_id, to_send,parse_mode="HTML")

    def detect_motion_old(self):

        initial_frame = self.frame[-1]
        sleep(self.delay)
        end_frame = self.frame[-1]

        if self.are_different(initial_frame, end_frame) and self.motion_flag:
            self.bot.sendMessage(self.send_id, "Movement detected")
            self.send_image(end_frame)
            sleep(5)

    def are_different(self, img1, img2, custom_score=0):

        if isinstance(img1, int) or isinstance(img2, int): return False
        similarity = self.get_similarity(img1, img2)
        if not custom_score:
            if similarity < self.diff_threshold:
                print(similarity,self.diff_threshold)
                return similarity

            else:
                return False
        else:
            if similarity < custom_score:
                print(similarity, custom_score)
                return similarity

            else:
                return False

    def denoise_img(self, image_list):

        print("denoising")

        if len(image_list)==1:
            denoised = cv2.fastNlMeansDenoisingColored(image_list[1], None, 10, 10, 7, 21)

        else:

            # make the list odd
            if (len(image_list)) % 2 == 0: image_list.pop()

            middle = int(float(len(image_list)) / 2 - 0.5)

            #getting smallest images size
            width=99999
            heigth=99999

            for img in image_list:
                size=tuple(img.shape[1::-1])
                if size[0]<width: width=size[0]
                if size[1]<heigth:heigth=size[1]

            #resizing all images to the smallest one
            image_list=[cv2.resize(elem,(width,heigth)) for elem in image_list]

            imgToDenoiseIndex = middle
            temporalWindowSize = len(image_list)
            hColor = 3
            #print(temporalWindowSize, imgToDenoiseIndex)

            denoised = cv2.fastNlMeansDenoisingColoredMulti(image_list, imgToDenoiseIndex, temporalWindowSize, hColor=hColor)
        print("denosed")

        return denoised

    def get_similarity(self, img1, img2):
        # start = datetime.now()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)
        # print("Convert to gray : " + str((datetime.now() - start).microseconds) + " microseconds")
        # start = datetime.now()
        # (score, diff) = compare_ssim(img1, img2, full=True)
        score = compare_psnr(img1, img2)
        # print("COMPAIRISON TIME : " + str((datetime.now() - start).microseconds) + " microseconds")

        # print(score)

        return score

    def send_image(self, img, msg=""):

        ret = cv2.imwrite(self.image_name, img)
        if not ret:
            self.bot.sendMessage(self.send_id, "There has been an error while writing the image")
            return

        with open(self.image_name, "rb") as file:
            if msg:
                self.bot.sendPhoto(self.send_id, file,caption=msg)
            else:
                self.bot.sendPhoto(self.send_id, file)


        os.remove(self.image_name)

    def send_video(self, video_name, msg=""):

        with open(video_name, "rb") as file:
            if msg: self.bot.sendMessage(self.send_id, msg)
            self.bot.sendVideo(self.send_id, file)
        os.remove(video_name)

    def detect_face(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img)
        if len(faces) > 0:
            # print("face detcted!")
            return faces

        return ()

    def face_on_video(self, frames):
        """This funcion add a rectangle on recognized faces"""

        colored_frames = []
        crop_frames = []
        faces=0

        # for every frame in the video
        for frame in frames:

            # detect if there is a face
            face = self.detect_face(frame)

            # if there is a face
            if len(face) > 0:
                # get the corners of the faces
                faces+=1
                for (x, y, w, h) in face:
                    # draw a rectangle around the corners
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

                    # if user want the face video too crop the image where face is detected
                    if self.face_photo_flag:
                        crop_frames.append(frame[y:y + h, x:x + w])

            # append colored frames
            colored_frames.append(frame)

        print(str(faces) + " frames with faces detected")


        return colored_frames, crop_frames
