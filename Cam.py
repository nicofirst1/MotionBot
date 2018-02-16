import copy
import threading
from multiprocessing import Pool
from threading import Thread
import os
import cv2
from time import sleep
from datetime import datetime
import logging

from utils import profiler

logger = logging.getLogger('motionlog')


class Cam_class:

    def __init__(self, bot):
        self.MAX_RETRIES = 4
        self.frames = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        self.telegram_handler = Telegram_handler(bot)
        self.telegram_handler.start()

        self.shotter = Cam_shotter(self.frames)
        self.shotter.start()

        self.motion = Cam_movement(self.shotter, self.telegram_handler)
        self.motion.start()
        logger.debug("Cam_class started")

    def capture_image(self, image_name):
        # print("taking image")
        img = self.frames[-2]

        if isinstance(img, int):
            print("empty queue")
            return False
        # try to save the image
        ret = cv2.imwrite(image_name, img)
        if ret:
            self.telegram_handler.send_image(image_name,"Camshot")
        else:
            self.telegram_handler.send_message("There has been an error while writing the image to file")



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

        self.telegram_handler.send_video(video_name,str(seconds)+" seconds record")


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
        self.camera_connected = False

        logger.debug("Cam_shotter started")

    def run(self):
        """Main thread loop"""

        while True:

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

    def __init__(self, shotter, telegram):
        # init the thread
        Thread.__init__(self)

        self.shotter = shotter
        self.frame = shotter.queue
        self.telegram_handler = telegram
        self.send_id = 24978334

        self.delay = 0.1
        self.diff_threshold = 0
        self.image_name = "different.png"
        self.min_area = 3000
        self.ground_frame = 0

        self.frontal_face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/lbpcascades/lbpcascade_frontalface.xml')


        self.profile_face_cascade = cv2.CascadeClassifier(
            '/home/pi/InstallationPackages/opencv-3.1.0/data/lbpcascades/lbpcascade_profileface.xml')

        self.max_seconds_retries = 10

        self.video_name = "detect_motion_video.mp4"
        self.resolution = (640, 480)  # width,height
        self.fps = 30
        self.out = cv2.VideoWriter(self.video_name, 0x00000021, self.fps, self.resolution)

        self.video_flag = True
        self.face_photo_flag = False
        self.motion_flag = True
        self.debug_flag = False

        self.resetting_ground = False

        logger.debug("Cam_movement started")

    def run(self):

        # wait for cam shotter to start
        while not self.shotter.camera_connected:
            sleep(0.5)

        # wait for the frame queue to be full
        initial_frame = self.frame[-1]
        while isinstance(initial_frame, int):
            initial_frame = self.frame[-1]

        # get the background image and save it
        self.reset_ground("Background image")

        while True:
            self.detect_motion_video()

    def detect_motion_video(self):

        # whait for resetting to be over
        while self.resetting_ground:
            continue

        # get end frame after delay seconds
        sleep(self.delay)
        end_frame = self.frame[-1]

        # calculate diversity
        score = self.are_different(self.ground_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames
        if self.motion_flag and score and not self.resetting_ground:

            logger.info("Movement detected")
            # send message
            self.motion_notifier(score)

            # do not capture video nor photo, just notification
            if not self.video_flag:
                sleep(3)
                return

            # create the file
            self.out.open(self.video_name, 0x00000021, self.fps, self.resolution)

            # start saving the frames
            self.shotter.capture(True)

            # self.send_image(initial_frame,"initial frame")
            # self.send_image(end_frame,"end frame")

            # while the current frame and the initial one are different (aka some movement detected)
            self.loop_difference(score, self.ground_frame, self.max_seconds_retries)

            # save the taken frames
            to_write = self.shotter.capture(False)

            # if the user wants the video of the movement
            if self.face_photo_flag:
                # take the face and send it
                to_write, face = self.face_on_video(to_write)

                if len(face) > 0:
                    self.telegram_handler.send_image(face, "Face found")
                else:
                    self.telegram_handler.send_message("Face not found")

            #write the movement on the video
            for elem in to_write:
                self.are_different(self.ground_frame, elem, True)


            # send the original video too
            if not self.resetting_ground:
                for elem in to_write:
                    cv2.rectangle(elem,(0,elem.shape[0]),( elem.shape[1], elem.shape[0]-10),(0,0,0),-1)
                    cv2.putText(elem, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                                (10, elem.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                    self.out.write(elem)
                self.out.release()
                self.telegram_handler.send_video(self.video_name)

            sleep(3)

    def reset_ground(self, msg):
        """function to reset the ground truth image"""
        self.resetting_ground = True
        gray = cv2.cvtColor(self.frame[-1], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self.ground_frame = gray
        self.telegram_handler.send_image(self.ground_frame, msg)
        self.resetting_ground = False

    def loop_difference(self, initial_score, initial_frame, seconds):
        """Loop until the current frame is the same as the ground image or time is exceeded"""

        start = datetime.now()
        end = datetime.now()
        score = initial_score
        print("Start of difference loop")
        while score and not self.resetting_ground:

            # take another fram
            prov = self.frame[-1]

            # check if images are different
            score = self.are_different(initial_frame, prov)

            # if time is exceeded exit while
            if (end - start).seconds > seconds:
                print("max seconds exceeded...checking for background changes")
                self.check_bk_changes(prov, 3)
                break

            # update current time in while loop
            end = datetime.now()

        print("End of difference loop")

    def check_bk_changes(self, initial_frame, seconds):
        #taking time
        start = datetime.now()
        end = datetime.now()
        #setting initial variables
        score = 0

        #setting initial frame
        gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        while not score:

            # take another frame
            prov = self.frame[-1]

            # check if images are different
            score = self.are_different(gray, prov)

            # if time is exceeded exit while
            if (end - start).seconds > seconds:
                print("max seconds exceeded")
                self.reset_ground("Back ground changed! New background")
                return True

            # update current time in while loop
            end = datetime.now()

        return False

    def are_different(self, grd_truth, img2, write_contour=False):
        # print("Calculation image difference")

        # blur and convert to grayscale
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # print(gray.shape, grd_truth.shape)

        # compute the absolute difference between the current frame and
        # first frame
        try:
            frameDelta = cv2.absdiff(grd_truth, gray)
        except cv2.error as e:
            error_log = "Cv Error: " + str(e) + "\ngrd_thruth : " + str(grd_truth.shape) + ", gray : " + str(
                gray.shape) + "\n"
            logger.error(error_log)
            print(error_log)
            return True

        thresh_original = cv2.threshold(frameDelta, 70, 255, cv2.THRESH_BINARY)[1]

        # self.send_image(frameDelta,"frameDelta")
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh_original, None, iterations=5)
        (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        found_area = False
        # print(len(cnts))
        for c in cnts:
            # if the contour is too small, ignore it
            # print("Area : "+str(cv2.contourArea(c)))
            if cv2.contourArea(c) < self.min_area:
                continue

            else:
                found_area = True
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                if write_contour:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if self.debug_flag:
                        self.telegram_handler.send_image(frameDelta)
                        self.telegram_handler.send_image(thresh_original, "Threshold Original")
                        self.telegram_handler.send_image(thresh, "Threshold Dilated")
                        self.telegram_handler.send_image(img2, "AREA: " + str(cv2.contourArea(c)))
            # print(found_area)

        return found_area

    # =========================FACE DETECION=======================================

    def denoise_img(self, image_list):

        print("denoising")

        if len(image_list) == 1:
            denoised = cv2.fastNlMeansDenoisingColored(image_list[0], None, 10, 10, 7, 21)

        else:

            # make the list odd
            if (len(image_list)) % 2 == 0: image_list.pop()

            middle = int(float(len(image_list)) / 2 - 0.5)

            # getting smallest images size
            width = 99999
            heigth = 99999

            for img in image_list:
                size = tuple(img.shape[1::-1])
                if size[0] < width: width = size[0]
                if size[1] < heigth: heigth = size[1]

            # resizing all images to the smallest one
            image_list = [cv2.resize(elem, (width, heigth)) for elem in image_list]

            imgToDenoiseIndex = middle
            temporalWindowSize = len(image_list)
            hColor = 3
            # print(temporalWindowSize, imgToDenoiseIndex)

            denoised = cv2.fastNlMeansDenoisingColoredMulti(image_list, imgToDenoiseIndex, temporalWindowSize,
                                                            hColor=hColor)
        print("denosed")

        return denoised

    def detect_face(self, img):

        scale_factor=1.4
        min_neight=3
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.frontal_face_cascade.detectMultiScale(img,scaleFactor=scale_factor,minNeighbors=min_neight)
        if len(faces) > 0:
            # print("face detcted!")
            return faces
        else:
            faces=self.profile_face_cascade.detectMultiScale(img,scaleFactor=scale_factor,minNeighbors=min_neight)
            if len(faces)> 0: return faces

        return ()

    def face_on_video_old(self, frames):


        new_frames=[]
        rectangular=[]
        faces=[]
        frame_count =0


        for frame in frames:
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            frame = frame[:, :, ::-1]
            frame_count += 1
            new_frames.append(frame)

    # Every 128 frames (the default batch size), batch process the list of frames to find faces
        batch_of_face_locations = face_recognition.batch_face_locations(new_frames, number_of_times_to_upsample=0)
        if len(new_frames) == 128:
            # Now let's list all the faces we found in all 128 frames
            for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
                number_of_faces_in_frame = len(face_locations)

                print("Found {} face(s)".format(number_of_faces_in_frame))

                for face_location in face_locations:
                    # Print the location of each face in this frame
                    x, y, w, h = face_location

                    #get the face cropped image
                    if self.face_photo_flag:
                        faces.append(frame[y:y + h, x:x + w])

                    #draw a rectangle around the face
                    frame= cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    rectangular.append(frame)

            new_frames=[]


        if len(faces)>0:
            face=self.denoise_img(faces)
        else: face=[]

        return rectangular,face


    def face_on_video(self, frames):
        """This funcion add a rectangle on recognized faces"""

        print("Face on video")

        colored_frames = []
        crop_frames = []
        faces = 0

        # for every frame in the video
        for frame in frames:

            # detect if there is a face
            face = self.detect_face(frame)

            # if there is a face
            if len(face) > 0:
                faces += 1
                # get the corners of the faces
                for (x, y, w, h) in face:

                    # if user want the face video too crop the image where face is detected
                    if self.face_photo_flag:
                        crop_frames.append(frame[y:y + h, x:x + w])

                    # draw a rectangle around the corners
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


            # append colored frames
            colored_frames.append(frame)

        if len(crop_frames)>0:

            face=self.denoise_img(crop_frames)

        else: face=()

        print(str(faces) + " frames with faces detected")

        return colored_frames, face




    # =========================TELEGRAM BOT=======================================

    def motion_notifier(self, score, degub=False):
        """Function to notify user dor a detected movement"""
        to_send = "Movement detected!\n"
        if degub:
            to_send += "Score is " + str(score) + "\n"

        if self.video_flag and not self.face_photo_flag:
            to_send += "<b>Video</b> is <b>ON</b>...it may take a minute or two"
        elif self.face_photo_flag and self.video_flag:
            to_send += "Both <b>Video</b> and <b>Face Photo</b> are <b>ON</b> ... it may take a while"

        self.telegram_handler.send_message(to_send, parse_mode="HTML")

    # =========================DEPRECATED=======================================

    def get_similarity(self, img1, img2):
        # start = datetime.now()
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img1 = cv2.equalizeHist(img1)
        img2 = cv2.equalizeHist(img2)
        # print("Convert to gray : " + str((datetime.now() - start).microseconds) + " microseconds")
        start = datetime.now()
        (score, diff) = compare_ssim(img1, img2, full=True, gaussian_weights=True)
        # score = compare_psnr(img1, img2)
        print("COMPAIRISON TIME : " + str((datetime.now() - start).microseconds) + " microseconds")

        print(score)

        return score

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

            foud_face = False
            self.shotter.capture(True)

            print("INITIAL SCORE : " + str(score))

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



class Telegram_handler(Thread):
    """Class to handle image/message/cideo sending throught telegram bot"""

    def __init__(self, bot):
        # init the thread
        Thread.__init__(self)

        self.bot = bot
        self.ids = self.get_ids(24978334)

        logger.info("Telegram handler started")

    def get_ids(self, fallback_id):
        """Get all the ids from the file"""
        # get ids form file
        ids_path = "Resources/ids"

        if "ids" in os.listdir("Resources/"):
            with open(ids_path, "r+") as file:
                lines = file.readlines()

            # every line has the id as the first element of a split(,)
            ids = []
            for id in lines:
                ids.append(int(id.split(",")[0]))
            return ids


        else:
            return [fallback_id]

    def send_image(self, img, msg=""):
        """Send an image to the ids """

        image_name = "image_to_send.png"

        ret = cv2.imwrite(image_name, img)

        if not ret:
            self.send_message("There has been an error while writing the image")
            return

        else:
            with open(image_name, "rb") as file:
                for id in self.ids:
                    if msg:
                        self.bot.sendPhoto(id, file, caption=msg)
                    else:
                        self.bot.sendPhoto(id, file)

        os.remove(image_name)
        logger.info("Image sent")

    def send_message(self, msg, parse_mode=""):

        for id in self.ids:
            self.bot.sendMessage(id, msg, parse_mode=parse_mode)

    def send_video(self, video_name, msg=""):

        if not video_name in os.listdir("."):
            self.send_message("The video could not be found ")
            return

        with open(video_name, "rb") as file:
            for id in self.ids:
                if msg: self.bot.sendVideo(id, file, caption=msg)
                else:self.bot.sendVideo(id, file)

        os.remove(video_name)
        logger.info("Video sent")

    def send_file(self,file_name,msg=""):

        if (file_name in os.listdir(".")):
            with open(file_name, "rb") as file:
                for id in self.ids:
                    if msg: self.bot.sendDocument(id, file,caption=msg)
                    else:  self.bot.sendDocument(id, file)

        else:
            self.send_message("No log file detected!")

