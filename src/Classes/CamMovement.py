import datetime
import logging
import sys
import threading
import traceback
from threading import Thread
from time import sleep

import cv2

# from memory_profiler import profile
from Classes.Darknet import Darknet

logger = logging.getLogger('cam_movement')


class CamMovement(Thread):
    """Class to detect movement from camera frames, recognize faces, movement direction and much more

    Attributes:

        shotter : the cam_shotter object

        telegram_handler : the telegram handler class

        face_recognizer: the face recognizer class

        delay : used delay between movement detection
        min_area : the minimum area of changes detected to be considered an actual change in the image
        ground_frame : the image used as the background to be compared with the current frames
        blur : the mean shift of the blur for the preprocessing of the images

        frontal_face_cascade : the object that detects frontal faces
        profile_face_cascade : the object that detects frontal faces
        face_size : the minimum window size to look for faces, the bigger the faster the program gets. But for distant
            people small values are to be taken into account

        max_seconds_retries : if a movement is detected for longer than max_seconds_retries the program will check for a
        background change, do not increase this parameter to much since it will slow down tremendously the program execution

        video_name : the name of the video to be send throught telegram
        resolution : the resolution of the video, do not change or telegram will not recognize the video as a gif
        fps : the frame per second of the video, higher values will result in slightly slower computation and more of a
            time loop video. Lower values will speed up the program (from 30 to 20 will give you a 25% speedup PER FUNCTION
            CALL) and will give the video a slower movement.


        motion_flag : flag used to check if the user want to recieve a notification (can be set by telegram)
        video_flag :  flag used to check if the user want to recieve a video of the movement (can be set by telegram)
        face_photo_flag : flag used to check if the user want to recieve a photo of the faces in the video (can be set by telegram)
        debug_flag : flag used to check if the user want to recieve the debug images (can be set by telegram , it slows down the program)
        face_reco_falg : flag used to check if the user want to recieve the predicted face with the photo (can be set by telegram)

        faces_cnts : list of contours for detected faces
        max_blurrines : the maximum threshold for blurriness detection, discard face images with blur>max_blurrines
        min_bk_threshold : the minimum difference in the background grayscaled image for the movement to be detected. When high
            only the bigger black/white difference will be detected. The range is (0,255) which is the intensity of the pixel

    """

    def __init__(self, shotter, telegram, face_recognizer,darknet):
        # init the thread
        Thread.__init__(self)
        self.stop_event = threading.Event()

        #classes
        self.shotter = shotter
        self.telegram_handler = telegram
        self.face_recognizer = face_recognizer
        self.darknet=darknet

        self.delay = 0.1
        self.min_area = 2000
        self.ground_frame = 0
        self.blur = (10, 10)


        self.max_seconds_retries = 10

        self.video_name = "detect_motion_video.mp4"
        self.resolution = (640, 480)  # width,height
        self.fps = 20

        self.video_flag = True
        self.face_photo_flag = False
        self.motion_flag = True
        self.debug_flag = False
        self.face_reco_falg = False
        self.green_squares_flag = False
        self.darknet_flag=True
        self.darknet_squares_flag=False

        self.resetting_ground = False

        self.faces_cnts = []
        self.max_blurrines = 100
        self.min_bk_threshold = 75
        self.dilate_window_size = (17, 13)

        logger.debug("Cam_movement started")

    def run(self):

        # wait for cam shotter to start
        while not self.shotter.camera_connected:
            sleep(0.5)

        # wait for the frame queue to be full
        initial_frame = self.shotter.get_gray_frame(-1)
        while isinstance(initial_frame, int):
            initial_frame = self.shotter.get_gray_frame(-1)

        # get the background image and save it
        self.reset_ground("Background image")

        while True:
            try:
                # detect a movement
                self.detect_motion_video()
            except:
                # on any exception log it and continue
                exc_type, exc_value, exc_traceback = sys.exc_info()
                lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                to_log = ''.join('!! ' + line for line in lines)
                logger.error(to_log)  # Log it
                print(to_log)

            if self.stopped():
                # if the thread has been stopped log and exit
                logger.info("Stopping Cam movement")
                return

    def stop(self):
        """Set the stop flag to true"""
        self.stop_event.set()

    def stopped(self):
        """Check for the flag value"""
        return self.stop_event.is_set()

    def detect_motion_video(self):
        """Principal function for this class
        It takes a new frame from the queue and check if it different from the ground image
        If it is capture the movement
        """

        # wait for resetting to be over
        while self.resetting_ground:
            continue

        # get end frame after delay seconds
        sleep(self.delay)
        end_frame = self.shotter.get_gray_frame(0)

        # calculate diversity
        score = self.are_different(self.ground_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames and the ground is not resetting
        if self.motion_flag and score and not self.resetting_ground:
            # start saving the frames
            self.shotter.capture(True)

            logger.info("Movement detected")
            # notify user
            self.motion_notifier(score)

            # do not capture video nor photo, just notification
            if not self.video_flag:
                self.shotter.capture(False)
                return

            # while the current frame and the initial one are different (aka some movement detected)
            self.loop_difference(score, self.ground_frame, self.max_seconds_retries)

            # save the taken frames
            to_write = self.shotter.capture(False)

            if self.darknet_flag:
                segmentation=self.darknet.detect_video(to_write)

                if self.darknet_squares_flag:
                    to_write=self.darknet.draw_bounds_list(segmentation)




            # # if the user wants the face in the movement
            # if self.face_photo_flag:
            #     # take the face
            #     # fixme
            #     face = None
            #     # if there are no faces found
            #     if face is None:
            #         self.telegram_handler.send_message(msg="Face not found")
            #
            #     else:
            #         for elem in face:
            #             self.telegram_handler.send_image(elem[2],
            #                                              msg="Found " + elem[0] + " with conficence = " + str(elem[1]))

            # send the original video too
            if not self.resetting_ground:
                print("Sending video...")
                # draw on the frames  and send video
                self.draw_on_frames(to_write)
                self.telegram_handler.send_video(self.video_name)
                print("...video sent")

    # =========================Movement=======================================

    def check_bk_changes(self, initial_frame, seconds):
        """Given the initial_frame and the max seconds the loop can run with, the function return false if movement is
        detected, otherwise, when the time has exceeded, it resets the back ground and return true"""
        # taking time
        start = datetime.datetime.now()
        end = datetime.datetime.now()
        # setting initial variables
        score = 0

        # setting initial frame
        # gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray = cv2.blur(initial_frame, self.blur, 0)

        # While there is movement
        while not score:

            # take another frame
            prov = self.shotter.get_gray_frame(0)

            # print(prov.shape)

            # check if images are different
            score = self.are_different(gray, prov)

            # if time is exceeded exit while
            if (end - start).seconds > seconds:
                print("max seconds exceeded")
                self.reset_ground("Back ground changed! New background")
                return True

            # update current time in while loop
            end = datetime.datetime.now()

        return False

    def loop_difference(self, initial_score, initial_frame, seconds, retry=False):
        """Loop until the current frame is the same as the ground image or time is exceeded, retry is used to
        be make this approach robust to tiny changes"""

        if retry: print("retriyng")
        # take the time
        start = datetime.datetime.now()
        end = datetime.datetime.now()
        # get the initial score
        score = initial_score
        print("Start of difference loop")

        # while there is some changes and the ground is not being resetted
        while score and not self.resetting_ground:

            # take the already grayscaled frame
            prov = self.shotter.get_gray_frame(0)
            # print(prov.shape)

            # check if images are different
            score = self.are_different(initial_frame, prov)

            # if time is exceeded exit while
            if (end - start).seconds > seconds:
                print("max seconds exceeded...checking for background changes")
                if not retry:
                    self.check_bk_changes(prov, 3)
                print("End of difference loop")
                return

            # update current time in while loop
            end = datetime.datetime.now()
            sleep(0.05)

        # it may be that there is no apparent motion
        if not retry:
            # wait a little and retry
            sleep(1.5)
            self.loop_difference(1, initial_frame, 1, True)

        print("End of difference loop")

    def are_different(self, grd_truth, img2):
        """Return whenever the difference in area between the ground image and the frame is grather than the
        threshold min_area."""

        cnts = self.compute_img_difference(grd_truth, img2)

        return any(cv2.contourArea(elem) > self.min_area for elem in cnts)

    def compute_img_difference(self, grd_truth, img2):
        """Compute te difference between the ground image and the frame passed as img2
        The ground image is supposed to be already preprocessed"""

        # blur and convert to grayscale
        # gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(img2, self.blur, 0)

        # compute the absolute difference between the current frame and
        # first frame
        try:
            frameDelta = cv2.absdiff(grd_truth, gray)
        except cv2.error as e:
            # catch any error and log
            error_log = "Cv Error: " + str(e) + "\ngrd_thruth : " + str(grd_truth.shape) + ", gray : " + str(
                gray.shape) + "\n"
            logger.error(error_log)
            print(error_log)
            # return true to not loose any movement
            return True

        # get the thresholded image
        thresh_original = cv2.threshold(frameDelta, self.min_bk_threshold, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.dilate_window_size)
        thresh = cv2.dilate(thresh_original, kernel, iterations=1)
        # get the contours of the changes
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if the debug flag is true send all the images
        if self.debug_flag:
            self.telegram_handler.send_image(frameDelta)
            self.telegram_handler.send_image(thresh_original, msg="Threshold Original")
            self.telegram_handler.send_image(thresh, msg="Threshold Dilated")
            self.telegram_handler.send_image(img2)

        # return the contours
        return cnts

    # =========================UTILS=======================================
    # @time_profiler()
    def draw_on_frames(self, frames, date=True):
        """Function to draw on frames"""

        face_color = (0, 0, 255)  # red
        motion_color = (0, 255, 0)  # green
        line_tickness = 2

        # create the file
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(self.video_name, fourcc, self.fps, self.resolution)
        out.open(self.video_name, fourcc, self.fps, self.resolution)

        prov_cnts = 0
        idx = 0
        face_idx = 0
        to_write = "Unkown - Unkown"
        print("Total frames to save : " + str(len(frames)))
        print("Total frames contours : " + str(len(self.faces_cnts)))
        for frame in frames:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.face_photo_flag:

                # take the corresponding contours for the frame
                # fixme
                face = None
                # face = self.faces_cnts[face_idx]
                face_idx += 1

                # if there is a face
                if face is not None:
                    # get the corners of the faces
                    for (x, y, w, h) in face:
                        # draw a rectangle around the corners
                        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, line_tickness)

            # draw movement
            if self.green_squares_flag:
                cnts = self.compute_img_difference(self.ground_frame, gray)

                # draw contours
                for c in cnts:
                    # if the contour is too small, ignore it
                    # print("Area : "+str(cv2.contourArea(c)))
                    if cv2.contourArea(c) < self.min_area:
                        pass

                    else:
                        # compute the bounding box for the contour, draw it on the frame,
                        # and update the text
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), motion_color, line_tickness)

                # add black rectangle at the bottom
                cv2.rectangle(frame, (0, frame.shape[0]), (frame.shape[1], frame.shape[0] - 30), (0, 0, 0), -1)

                # write the movement direction
                if not idx % 2:
                    prov_cnts = cnts
                elif len(prov_cnts) > 0 and len(cnts) > 0:
                    movement, _ = self.movement_direction(prov_cnts, cnts)
                    to_write = ""
                    if movement[0]:
                        to_write += "Incoming - "

                    else:
                        to_write += "Outgoing - "

                    if movement[1]:
                        to_write += "Left"

                    else:
                        to_write += "Right"

                    # cv2.circle(frame, center_points[0], 1, (255, 255, 0), 10,2)
                    # cv2.circle(frame, center_points[1], 1, (255, 255, 255), 10,2)

                cv2.putText(frame, to_write,
                            (frame.shape[1] - 250, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 255),
                            1)

                idx += 1

            # add a date to the frame
            if date:
                # write time
                correct_date = datetime.datetime.now() + datetime.timedelta(hours=1)

                cv2.putText(frame, correct_date.strftime("%A %d %B %Y %H:%M:%S"),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

            # write frames on file
            out.write(frame)

        # empty the face contours list
        self.faces_cnts = []

        # free file
        out.release()

    def reset_ground(self, msg):
        """Reset the ground truth image"""

        print("Reset ground image ...")
        # set the flag
        self.resetting_ground = True

        # convert to gray and blur
        gray = self.shotter.get_gray_frame(-1)
        gray = cv2.blur(gray, self.blur, 0)
        # set the frame and notify
        self.ground_frame = gray
        self.telegram_handler.send_image(self.ground_frame, msg=msg)

        self.resetting_ground = False

        logger.info("Ground image reset")
        print("Done")

    @staticmethod
    def movement_direction(cnts1, cnts2):
        """Function to get the movement direction from two frames
        returns a tuple where the first elem is the outgoing/incoming 0/1, the second is right/left 0/1"""

        # the incoming outgoing movement is estimated throught the sum of the areas
        # if sum(areas1)<sum(areas2) the object is outgoing

        area1 = sum(cv2.contourArea(c) for c in cnts1)
        area2 = sum(cv2.contourArea(c) for c in cnts2)

        # print(area1,area2)

        # the left/right is given by the position of the averaged center of each area
        # if center1>center2 the object is moving right to left (so left)
        # get the centers of each area on the x axis
        centers1 = []
        centers2 = []

        # get the area
        for c in cnts1:
            centers1.append(cv2.boundingRect(c))

        # calculate the center point
        center_point1 = [((x + (x + w)) / 2, (y + (y + h)) / 2) for (x, y, w, h) in centers1]
        # calculate the mean position on the x axis
        centers1 = [(x + (x + w)) / 2 for (x, y, w, h) in centers1]

        # get the area
        for c in cnts2:
            centers2.append(cv2.boundingRect(c))

        center_point2 = [((x + (x + w)) / 2, (y + (y + h)) / 2) for (x, y, w, h) in centers2]
        # calculate the mean position on the x axis
        centers2 = [(x + (x + w)) / 2 for (x, y, w, h) in centers2]

        # avarage the center
        centers1 = sum(centers1) / len(centers1)
        centers2 = sum(centers2) / len(centers2)

        # avarage the center
        center_point1 = (int(sum(elem[0] for elem in center_point1) / len(center_point1)),
                         int(sum(elem[1] for elem in center_point1) / len(center_point1)))
        center_point2 = (int(sum(elem[0] for elem in center_point2) / len(center_point2)),
                         int(sum(elem[1] for elem in center_point2) / len(center_point2)))

        # print(center_point1,center_point2)

        return (area1 < area2, centers1 > centers2), (center_point1, center_point2)




    # =========================TELEGRAM BOT=======================================

    def motion_notifier(self, score, degub=False):
        """Function to notify user for a detected movement"""
        to_send = "Movement detected!\n"
        if degub:
            to_send += "Score is " + str(score) + "\n"

        if self.face_photo_flag:
            to_send+="<b>Face Photo</b>, "

        if self.face_reco_falg:
            to_send+="<b>Face Reco</b>, "

        if self.video_flag:
            to_send+="<b>Video</b>, "

        if self.darknet_flag:
            to_send+="<b>Darknet</b>, "

        to_send+="are  <b>ON</b>...it may take a minute or two"



        self.telegram_handler.send_message(to_send, parse_mode="HTML")

    def send_ground(self, specific_id, msg):
        """Send the ground image to the users"""
        self.telegram_handler.send_image(self.ground_frame, specific_id=specific_id, msg=msg)

    # =========================DEPRECATED=======================================

    def detect_motion_photo(self):
        initial_frame = self.shotter.get_gray_frame(-1)

        sleep(self.delay)
        end_frame = self.shotter.get_gray_frame(-1)

        # calculate diversity
        score = self.are_different(initial_frame, end_frame)
        # if the notification is enable and there is a difference between the two frames
        if self.motion_flag and score:

            # send message
            self.motion_notifier(score)

            # take a new (more recent) frame
            prov = self.shotter.get_gray_frame(-1)

            # take the time
            start = datetime.datetime.now()
            end = datetime.datetime.now()

            foud_face = False
            self.shotter.capture(True)

            print("INITIAL SCORE : " + str(score))

            # while the current frame and the initial one are different (aka some movement detected)
            while self.are_different(initial_frame, prov):

                print("in while")
                # check for the presence of a face in the frame
                # if self.detect_face(prov):
                #     # if face is detected send photo and exit while
                #     self.telegram_handler.send_image(prov, msg="Face detected!")
                #     break

                # take another frame
                prov = self.shotter.get_gray_frame(-1)

                # if time is exceeded exit while
                if (end - start).seconds > self.max_seconds_retries:
                    print("max seconds exceeded")
                    break

                # update current time in while loop
                end = datetime.datetime.now()

            if not foud_face:
                self.telegram_handler.send_image(end_frame, msg="Face not detected")
            sleep(3)

    @staticmethod
    def denoise_img(image_list):
        """Denoise one or multiple images"""

        print("denoising")

        if len(image_list) == 1:
            denoised = cv2.fastNlMeansDenoisingColored(image_list[0], None, 10, 10, 7, 21)

        else:

            # make the list odd
            if (len(image_list)) % 2 == 0:
                image_list.pop()
            # get the middle element
            middle = int(float(len(image_list)) / 2 - 0.5)

            width = sys.maxsize
            heigth = sys.maxsize

            # getting smallest images size
            for img in image_list:
                size = tuple(img.shape[1::-1])
                if size[0] < width: width = size[0]
                if size[1] < heigth: heigth = size[1]

            # resizing all images to the smallest one
            image_list = [cv2.resize(elem, (width, heigth)) for elem in image_list]

            imgToDenoiseIndex = middle
            temporalWindowSize = len(image_list)
            hColor = 3
            searchWindowSize = 17
            hForColorComponents = 1
            # print(temporalWindowSize, imgToDenoiseIndex)

            denoised = cv2.fastNlMeansDenoisingColoredMulti(image_list, imgToDenoiseIndex, temporalWindowSize,
                                                            hColor=hColor, searchWindowSize=searchWindowSize,
                                                            hForColorComponents=hForColorComponents)
        print("denosed")

        return denoised
