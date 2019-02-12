import logging
import os
from threading import Thread

import cv2

# from memory_profiler import profile

logger = logging.getLogger('telegram_handler')


class TelegramHandler(Thread):
    """Class to handle image/message/video sending throught telegram bot"""

    def __init__(self, bot):
        # init the thread
        Thread.__init__(self)

        self.bot = bot
        self.default_id = 24978334
        self.ids = self.get_ids(self.default_id)

        # print(self.ids)

        logger.info("Telegram handler started")

    @staticmethod
    def get_ids(fallback_id):
        """Get all the ids from the file"""
        # get ids form file
        print("getting ids from file")
        ids_path = "Resources/ids"

        # if there are some ids in the file get them
        if "ids" in os.listdir("Resources/"):
            with open(ids_path, "r+") as file:
                lines = file.readlines()

            # every line has the id as the first element of a split(,)
            ids = []
            for user_id in lines:
                if int(user_id.split(",")[1]):
                    ids.append(int(user_id.split(",")[0]))
            return ids

        else:
            # return the default id
            return [fallback_id]

    def send_image(self, img, specific_id=0, msg=""):
        """Send an image to the ids """

        image_name = "image_to_send.png"

        ret = cv2.imwrite(image_name, img)

        if not ret and specific_id:
            self.send_message("There has been an error while writing the image", specific_id=specific_id)
            return
        elif not ret:
            if not ret and specific_id:
                self.send_message("There has been an error while writing the image")
                return

        else:
            with open(image_name, "rb") as file:
                if not specific_id:
                    for user_id in self.ids:
                        if msg:
                            self.bot.sendPhoto(user_id, file, caption=msg)
                        else:
                            self.bot.sendPhoto(user_id, file)
                else:
                    if msg:
                        self.bot.sendPhoto(specific_id, file, caption=msg)
                    else:
                        self.bot.sendPhoto(specific_id, file)

        try:
            os.remove(image_name)
        except FileNotFoundError:
            pass

        logger.info("Image sent")

    def send_message(self, msg, specific_id=0, parse_mode=""):
        """Send a message to the ids"""

        if not specific_id:
            for user_id in self.ids:
                self.bot.sendMessage(user_id, msg, parse_mode=parse_mode)
        else:
            self.bot.sendMessage(specific_id, msg, parse_mode=parse_mode)

    def send_video(self, video_name, specific_id=0, msg=""):
        """Send a video to the ids"""

        try:
            with open(video_name, "rb") as file:
                if not specific_id:
                    for user_id in self.ids:
                        if msg:
                            self.bot.sendVideo(user_id, file, caption=msg)
                        else:
                            self.bot.sendVideo(user_id, file)
                else:
                    if msg:
                        self.bot.sendVideo(specific_id, file, caption=msg)
                    else:
                        self.bot.sendVideo(specific_id, file)

            os.remove(video_name)

        except FileNotFoundError:
            self.send_message("The video could not be found ", specific_id=specific_id)

        logger.info("Video sent")

    def send_file(self, file_name, specific_id=0, msg=""):
        """Send a file to the ids"""

        if file_name in os.listdir("."):
            with open(file_name, "rb") as file:
                if not specific_id:
                    for user_id in self.ids:
                        if msg:
                            self.bot.sendDocument(user_id, file, caption=msg)
                        else:
                            self.bot.sendDocument(user_id, file)
                else:
                    if msg:
                        self.bot.sendDocument(specific_id, file, caption=msg)
                    else:
                        self.bot.sendDocument(specific_id, file)
        else:
            self.send_message("No log file detected!", specific_id=specific_id)
