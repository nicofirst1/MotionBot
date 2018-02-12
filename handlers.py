# coding=utf-8
from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ConversationHandler, Updater

import os

from Cam import Cam_class
from utils import add_id, elegible_user, read_token_psw


TOKEN,psw=read_token_psw()
print("TOKEN : "+TOKEN+"\nPassword : "+psw)


updater = Updater(TOKEN)
disp = updater.dispatcher

cam = Cam_class(updater.bot)


FLAG_KEYBOARD= InlineKeyboardMarkup([
    [InlineKeyboardButton("Motion Detection", callback_data="/flag motion"),
    InlineKeyboardButton("Face Video", callback_data="/flag face_video"),
     InlineKeyboardButton("Face Photo", callback_data="/flag face_photo")],
    [InlineKeyboardButton("Debug", callback_data="/flag debug"),
     InlineKeyboardButton("Done", callback_data="/flag done")]

])

FLAG_SEND="""
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

    global FLAG_KEYBOARD

    #print("Flag Main")

    to_send=complete_flags()
    update.message.reply_text(to_send,reply_markup=FLAG_KEYBOARD,parse_mode="HTML")


def complete_flags():
    global FLAG_SEND

    complete_falg_str=FLAG_SEND

    # get falg values
    motion_detection = cam.motion.motion_flag
    face_v = cam.motion.faces_video_flag
    face_p = cam.motion.face_photo_flag
    debug=cam.motion.debug_flag

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


def flag_setting_callback(bot,update):
    param = update.callback_query.data.split()[1]

    global FLAG_KEYBOARD

    #print("Flag callback")


    if param=="motion":
        cam.motion.motion_flag=not cam.motion.motion_flag
        if not cam.motion.motion_flag:
            cam.motion.face_photo_flag=False
            cam.motion.faces_video_flag=False
    elif param=="face_video":
        cam.motion.faces_video_flag=not cam.motion.faces_video_flag
    elif param=="face_photo":
        cam.motion.face_photo_flag=not cam.motion.face_photo_flag

    elif param=="debug":
        cam.motion.debug_flag=not cam.motion.debug_flag

    elif param=="done":
        bot.delete_message(
            chat_id=update.callback_query.message.chat_id,
            message_id=update.callback_query.message.message_id
        )
        return


    to_change=complete_flags()




    bot.edit_message_text(
            chat_id=update.callback_query.message.chat_id,
            text=to_change,
            message_id=update.callback_query.message.message_id,
            parse_mode="HTML",
            reply_markup=FLAG_KEYBOARD
        )



def start(bot, update):
    #print("start")
    update.message.reply_text("Welcome... to start insert the password")
    return 1



def get_psw(bot, update):
    user_psw = update.message.text

    if not user_psw == psw:
        update.message.reply_text("Incorrect password...you can not accesst this bot functionalities anymore :(")
        add_id(update.message.from_user.id,0)
    else:
        update.message.reply_text("Correct password!")
        add_id(update.message.from_user.id,1)



def annulla(bot, update):
    update.message.reply_text("Error")
    return ConversationHandler.END

@elegible_user
def get_camshot(bot, update):
    image = "image.png"
    ret=cam.capture_image(image)


    if ret:
        with open(image, "rb") as file:
            bot.sendPhoto(update.message.from_user.id, file)
        os.remove(image)
    else:
        update.message.reply_text("There has been an error...please retry in a few seconds")


@elegible_user
def stream(bot, update,args):
    print("Video")
    max_seconds=20
    if not args:
        SECONDS=5
    else:
        if not len(args)==1:
            update.message.reply_text("You must provide just ONE number for the seconds")
            return
        try:
            SECONDS=int(args[0])
        except ValueError:
            update.message.reply_text("You did not provide aright number")
            return

        if SECONDS>max_seconds:
            update.message.reply_text("The maximum seconds is "+str(max_seconds)+"...setting deafult 5s")
            SECONDS=5

    video_name="video.mp4"

    update.message.reply_text("Wait "+str(SECONDS)+" seconds...")

    cam.capture_video(video_name,SECONDS)

    print("Capture complete")


    with open(video_name, "rb") as file:
        bot.sendVideo(update.message.from_user.id, file)
    os.remove(video_name)

