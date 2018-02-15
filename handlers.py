# coding=utf-8
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ConversationHandler, Updater
import logging

import os,sys

from Cam import Cam_class
from utils import add_id, elegible_user, read_token_psw

TOKEN, psw = read_token_psw()
print("TOKEN : " + TOKEN + "\nPassword : " + psw)

updater = Updater(TOKEN)
disp = updater.dispatcher
logger = logging.getLogger('motionlog')

cam = Cam_class(updater.bot)

FLAG_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("Motion Detection", callback_data="/flag motion"),
     InlineKeyboardButton("Face Video", callback_data="/flag face_video"),
     InlineKeyboardButton("Face Photo", callback_data="/flag face_photo")],
    [InlineKeyboardButton("Debug", callback_data="/flag debug"),
     InlineKeyboardButton("Done", callback_data="/flag done")]

])

FLAG_SEND = """
Here you can set the values of your flags, either <b>ON</b> or <b>OFF</b>
-- <b>Motion Detection</b> : If set to <i>ON</i> the bot will notify, both with a message and with a video, you when a movement has been detected
---- <b>Face Video</b> : If set to <i>ON</i> the video you recieve from the <i>Motion Detection</i> above will highlith faces
---- <b>Face Photo</b> : If set to <i>ON</i> you will recieve a photo of the detected face with the video
-- <b>Debug</b> : If set to <i>ON</i> you will recieve the images from the debug
To set a flag just click on the corrispondent button.
Note that <b>Face Photo</b> depends on  <b>Face Video</b> which depends on <b>Motion Detection</b>, so unless this last on is set <b>ON</b> the other won't work
Current value are the following :"""


@elegible_user
def flag_setting_main(bot, update):
    """Telegram command to set the flags for the motion detection"""
    global FLAG_KEYBOARD

    # print("Flag Main")

    to_send = complete_flags()
    update.message.reply_text(to_send, reply_markup=FLAG_KEYBOARD, parse_mode="HTML")


@elegible_user
def reset_ground(bot,update):
    """Telegram command to reset the ground truth image (the background)"""

    cam.motion.reset_ground()
    update.message.reply_text("Ground image has been reset")


def complete_flags():
    """Function to return the changed flag text"""
    global FLAG_SEND

    complete_falg_str = FLAG_SEND

    # get falg values
    motion_detection = cam.motion.motion_flag
    face_v = cam.motion.faces_video_flag
    face_p = cam.motion.face_photo_flag
    debug = cam.motion.debug_flag

    complete_falg_str += "\n-- <b>Motion Detection</b>"

    # complete message
    if motion_detection:
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Face Video</b>"

    if face_v:
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Face Photo</b>"

    if face_p:
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Debug</b>"

    if debug:
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    return complete_falg_str


def flag_setting_callback(bot, update):
    """Function to respond to the user choiche for the flag setting"""
    param = update.callback_query.data.split()[1]

    global FLAG_KEYBOARD

    # print("Flag callback")

    if param == "motion":
        cam.motion.motion_flag = not cam.motion.motion_flag
        if not cam.motion.motion_flag:
            cam.motion.face_photo_flag = False
            cam.motion.faces_video_flag = False
    elif param == "face_video":
        cam.motion.faces_video_flag = not cam.motion.faces_video_flag
    elif param == "face_photo":
        cam.motion.face_photo_flag = not cam.motion.face_photo_flag

    elif param == "debug":
        cam.motion.debug_flag = not cam.motion.debug_flag

    elif param == "done":
        bot.delete_message(
            chat_id=update.callback_query.message.chat_id,
            message_id=update.callback_query.message.message_id
        )
        return

    to_change = complete_flags()

    bot.edit_message_text(
        chat_id=update.callback_query.message.chat_id,
        text=to_change,
        message_id=update.callback_query.message.message_id,
        parse_mode="HTML",
        reply_markup=FLAG_KEYBOARD
    )


def start(bot, update):
    """Telegram command to start the bot ( it takes part of the conversation handler)"""
    # print("start")
    update.message.reply_text("Welcome... to start insert the password")
    return 1


def annulla(bot, update):
    """Fallback function for the conversation handler"""
    update.message.reply_text("Error")
    return ConversationHandler.END


@elegible_user
def get_camshot(bot, update):
    """Telegram command to get a camshot from the camera"""

    logger.info("/Photo command called")

    image = "image.png"
    cam.capture_image(image)




@elegible_user
def stream(bot, update, args):
    """Telegram command to take a video from the camera"""
    print("Video")
    logger.info("/Video command called")

    max_seconds = 20
    if not args:
        SECONDS = 5
    else:
        if not len(args) == 1:
            update.message.reply_text("You must provide just ONE number for the seconds")
            return
        try:
            SECONDS = int(args[0])
        except ValueError:
            update.message.reply_text("You did not provide aright number")
            return

        if SECONDS > max_seconds:
            update.message.reply_text("The maximum seconds is " + str(max_seconds) + "...setting deafult 5s")
            SECONDS = 5

    video_name = "video.mp4"

    update.message.reply_text("Wait " + str(SECONDS) + " seconds...")

    cam.capture_video(video_name, SECONDS)


@elegible_user
def stop_execution(bot, update):
    """Telegram command to stop the bot execution """

    logger.info("Stopping execution")
    update.message.reply_text("Stopping surveillance")
    sys.exit(0)

@elegible_user
def send_log(bot,update):
    """Telegram command to send the logger file"""

    cam.telegram_handler.send_file("motion.log","logs")



@elegible_user
def send_ground(bot, update):
    image_name="groung.png"

    ret = cv2.imwrite(image_name, cam.motion.ground_frame)
    if not ret:
        cam.telegram_handler.send_message( "There has been an error while writing the image")
        return

    cam.telegram_handler.send_image(image_name,"Current BackGround image")

