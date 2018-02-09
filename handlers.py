# coding=utf-8

from telegram.ext import ConversationHandler

import os

import Cam
from utils import add_id, elegible_user

psw = "SuperMegaFamBrand123!"
cam= Cam.Cam_class()


def start(bot, update):
    print("start")
    update.message.reply_text("Benvenuto...per iniziare inserisci la password")
    return 1



def get_psw(bot, update):
    user_psw = update.message.text

    if not user_psw == psw:
        update.message.reply_text("Password incorretta...non ti è stato garantito l'accesso :(")
        add_id(update.message.from_user.id,0)
    else:
        update.message.reply_text("Password corretta...ti è stato garantito l'accesso al bot")
        add_id(update.message.from_user.id,1)



def annulla(bot, update):
    update.message.reply_text("annullo")
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
        update.message.reply_text("Si è verificato un errore...riprova")


@elegible_user
def stream(bot, update):
    print("Video")
    SECONDS=5
    video_name="video.avi"

    update.message.reply_text("Attendi "+str(SECONDS)+" secondi...")

    cam.capture_video(video_name,SECONDS)

    print("Capture complete")


    with open(video_name, "rb") as file:
        print("file opened")
        bot.sendVideo(update.message.from_user.id, file)
    os.remove(video_name)



