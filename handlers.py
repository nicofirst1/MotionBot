# coding=utf-8

from telegram.ext import ConversationHandler, Updater

import os

from Cam import Cam_class
from utils import add_id, elegible_user, read_token_psw


TOKEN,psw=read_token_psw()
print("TOKEN : "+TOKEN+"\nPassword : "+psw)


updater = Updater(TOKEN)
disp = updater.dispatcher
cam = Cam_class(updater.bot)



def start(bot, update):
    print("start")
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



@elegible_user
def notification(bot, update, args):
    if not args:
        update.message.reply_text("You have not specified ON/OFF")
        return

    if args[0].lower()=="on":
        cam.motion.notification=True
        update.message.reply_text("Notification enabled")


    elif args[0].lower()=="off":
        cam.motion.notification=False
        update.message.reply_text("Notification disabled")

    else:
        update.message.reply_text("You must use this command followed by ON/OFF")


def face_detection(bot,update, args):
    if not args:
        update.message.reply_text("You have not specified ON/OFF")
        return

    if args[0].lower() == "on":
        cam.motion.get_faces = True
        update.message.reply_text("Face detection enabled")


    elif args[0].lower() == "off":
        cam.motion.get_faces = False
        update.message.reply_text("Face detection disabled")

    else:
        update.message.reply_text("You must use this command followed by ON/OFF")

