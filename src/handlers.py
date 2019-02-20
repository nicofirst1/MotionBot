# coding=utf-8
import logging
import os
import sys
import urllib.request
import telegram
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ConversationHandler, Updater

from src.Classes.Main_class import MainClass
from src.Utils.utils import add_id, elegible_user, read_token_psw
from Path import Path as pt


TOKEN, psw = read_token_psw()
print("TOKEN : " + TOKEN + "\nPassword : " + psw)

updater = Updater(TOKEN)
disp = updater.dispatcher
logger = logging.getLogger('motionlog')

cam = MainClass(updater)

FLAG_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("Motion Detection", callback_data="/flag motion"),
     InlineKeyboardButton("Video", callback_data="/flag video"),
     InlineKeyboardButton("Movement Square", callback_data="/flag square")],

    [InlineKeyboardButton("Face Photo", callback_data="/flag face_photo"),
     InlineKeyboardButton("Face Reco", callback_data="/flag face_reco"),
     InlineKeyboardButton("Debug", callback_data="/flag debug")],
    [InlineKeyboardButton("Darknet", callback_data="/flag darknet"),
     InlineKeyboardButton("Done", callback_data="/flag done")],

])

FLAG_SEND = """
Here you can set the values of your flags, either <b>ON</b> or <b>OFF</b>
-- <b>Motion Detection</b> : If set to <i>ON</i> the bot will notify, both with a message and with a video, you when a movement has been detected
---- <b>Video</b> : If set to <i>ON</i> the video you recieve from the <i>Motion Detection</i> above will highlith faces
---- <b>Movement square</b> : If set to <i>ON</i> the video will present green squares where the movement has been detected, (set it to off for faster computation)
---- <b>Face Photo</b> : If set to <i>ON</i> you will recieve a photo of the detected face with the video
---- <b>Face Reco(gnizer)</b> : If set to <i>ON</i> the program will try to guess the person face
---- <b>Darknet</b> : If set to <i>ON</i> the program will use darknet for object segmentation
-- <b>Debug</b> : If set to <i>ON</i> you will recieve the images from the debug

To set a flag just click on the corrispondent button.
Note that <b>Face Photo</b> depends on  <b>Face Video</b> which depends on <b>Motion Detection</b>, so unless this last on is set <b>ON</b> the other won't work
Current value are the following :"""


# ===============Callback, Conversation===================
def annulla(bot, update):
    """Fallback function for the conversation handler"""
    update.message.reply_text("Error")
    return ConversationHandler.END


def flag_setting_callback(bot, update):
    """Function to respond to the user choiche for the flag setting"""
    param = update.callback_query.data.split()[1]

    global FLAG_KEYBOARD

    # print("Flag callback")

    if param == "motion":
        cam.motion.flags.flip_value('motion')

    elif param == "video":
        cam.motion.flags.flip_value('video')


    elif param == "square":
        cam.motion.flags.flip_value('green squares')


    elif param == "face_photo":
        cam.motion.flags.flip_value('face photo')

    elif param == "debug":
        cam.motion.flags.flip_value('debug')


    elif param == "face_reco":
        cam.motion.flags.flip_value('face reco')



    elif param == "darknet":
        cam.motion.flags.flip_value('darknet')



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


def get_psw(bot, update):
    user_psw = update.message.text

    if not user_psw == psw:
        update.message.reply_text("Incorrect password...you can not accesst this bot functionalities anymore :(")
        add_id(update.message.from_user.id, 0)
    else:
        update.message.reply_text("Correct password!")
        add_id(update.message.from_user.id, 1)

    cam.telegram_handler.get_ids(cam.telegram_handler.default_id)


# ===============Commands===================

@elegible_user
def flag_setting_main(bot, update):
    """Telegram command to set the flags for the motion detection"""
    global FLAG_KEYBOARD

    # print("Flag Main")

    to_send = complete_flags()
    update.message.reply_text(to_send, reply_markup=FLAG_KEYBOARD, parse_mode="HTML")


@elegible_user
def reset_ground(bot, update):
    """Telegram command to reset the ground truth image (the background)"""
    update.message.reply_text("Resetting ground image")

    username = update.message.from_user.username
    cam.motion.reset_ground("Reset ground asked from @" + username)
    update.message.reply_text("Ground image has been reset")


@elegible_user
def get_camshot(bot, update):
    """Telegram command to get a camshot from the camera"""
    image = "image_" + str(update.message.from_user.id) + ".png"
    ret = cam.capture_image(image)
    logger.info("photo command called")

    if ret:
        with open(image, "rb") as file:
            bot.sendPhoto(update.message.from_user.id, file)
        os.remove(image)
    else:
        update.message.reply_text("There has been an error...please retry in a few seconds")


@elegible_user
def stream(bot, update, args):
    """Telegram command to take a video from the camera"""
    print("Video")
    logger.info("video command called")

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

    video_name = "video_" + str(update.message.from_user.id) + ".mp4"

    update.message.reply_text("Wait " + str(SECONDS) + " seconds...")

    cam.capture_video(video_name, SECONDS, update.message.from_user.id)

    logger.info("Sending a " + str(SECONDS) + " seconds video")
    print("Capture complete")


@elegible_user
def stop_execution(bot, update):
    """Telegram command to stop the bot execution """
    logger.info("stop command called")

    cam.telegram_handler.send_message("Stopping surveillance")
    cam.stop()
    logger.info("Stopping execution")
    sys.exit(0)


@elegible_user
def send_log(bot, update):
    """Telegram command to send the logger file"""
    logger.info("send log command called")

    sent=False

    for log in os.listdir(pt.LOGGER_DIR):
        if ".log" in log:
            with open(f"{pt.LOGGER_DIR}/{log}", "rb") as file:
                try:
                    bot.sendDocument(update.message.chat_id, file)
                    sent=True
                except telegram.error.BadRequest:
                    pass
    if not sent:
        update.message.reply_text("No log file detected!")


@elegible_user
def delete_log(bot, update):
    """Telegram command to send the logger file"""
    logger.info("delete log command called")


    delete=False

    for log in os.listdir(pt.LOGGER_DIR):
        if ".log" in log:
            os.remove(f"{pt.LOGGER_DIR}/{log}")
            delete=True

    if not delete:
        update.message.reply_text("No log file detected!")
    else:
        update.message.reply_text("Log files deleted")





@elegible_user
def send_ground(bot, update):
    logger.info("ground command called")

    print("Sending ground...")

    cam.motion.send_ground(update.message.from_user.id, "Current background image")

    print("...Done")


@elegible_user
def help_bot(bot, update):
    help_str = """
Welcome to this bot!
You can use it to with a camera to create your own surveillance system.
The avaiable commands are the following:
- /start - start bot
- /photo : get a camshot from the camera
- /video seconds : get a video from the camera with <i>seconds</i> duration, the default duration is 5 seconds
- /flags : set the flags 
- /resetg : reset the backgroud image 
- /stop : stop surveillance execution
- /logsend : send the logger file
- /logdel : delete the log file
- /bkground : send the background image
- /classify : classify the person face

This bot has multiple functionalities:
<b>==Movement detection==</b>
When there is a detected change between the background image and the current frame from the camera you will be notified with a message. 
You can set the flags (try the /flags command) to get multiple information from the camera, such as the video of the movement, the detected faces in the video and so on.

<b>==Camera shotter==</b>
You can use the commands /photo and /video to get a <b>live</b> update of what the camera is seeing

<b>==Telegram access==</b>
When a new user starts the bot it will be asked for the password. This password can be set from the source code <b>ONLY</b> and you will have <b>just one chance</b> to get it right.
If you fail the bot will block you and no commands will be executed. Otherwise you will be granted full access to the bot functionalities.
The bot password can be set in the <i>Resources/token_psw.txt</i> file (check out the README).
When a movement is detected, depending on the flags values, every id in the ids file will be notified (if not blocked).

<b>==Face recognizer==</b>
If the proper falg is set to True and a face has been detected in the video, the face recognizer will try to guess the person whose face has been seen.
When the bot first starts it will train the face recognizer with the saved faces. To save a new faces simply use the /classify command and follow th instructions.
The more faces the recognizer find out the more precise it will be

Thank you for choosing this bot, I hope you like it.
"""
    update.message.reply_text(help_str, parse_mode="HTML")


@elegible_user
def predict_face(bot, update):
    """
    Predict the face in the photo
    :param bot:
    :param update:
    :return:
    """

    file_path = "downloaded.jpg"
    # get the photo file
    fileID = update.message.photo[-1].file_id
    file_url = bot.get_file(fileID).file_path

    urllib.request.urlretrieve(file_url, file_path)

    # get the text to send
    img=cam.predict_face(file_path)
    if img is None:
        update.message.reply_text("Sorry...no faces found", parse_mode="HTML")
        return

    cam.telegram_handler.send_image(img, update.message.from_user.id)


# ===============Utils===================


def complete_flags():
    """Function to return the changed flag text"""
    global FLAG_SEND

    complete_falg_str = FLAG_SEND



    complete_falg_str += "\n-- <b>Motion Detection</b>"

    # complete message
    if cam.motion.flags.get_value('motion'):
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Video</b>"

    if cam.motion.flags.get_value('video'):
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Movement square</b>"

    if cam.motion.flags.get_value('green squares'):
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Face Photo</b>"

    if cam.motion.flags.get_value('face photo'):
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Face Reco</b>"

    if cam.motion.flags.get_value('face reco'):
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    complete_falg_str += "\n-- <b>Darknet</b>"

    if cam.motion.flags.get_value('darknet'):
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"


    complete_falg_str += "\n-- <b>Debug</b>"

    if cam.motion.flags.get_value('debug'):
        complete_falg_str += " ✅"
    else:
        complete_falg_str += " ❌"

    return complete_falg_str
