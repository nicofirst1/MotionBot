# coding=utf-8

from time import sleep
from telegram.ext import ConversationHandler
from subprocess import call
import cv2
import os

from utils import  add_id, elegible_user

MAX_RETRIES = 8
psw = "SuperMegaFamBrand123!"
CAM=cv2.VideoCapture(0)


def start(bot, update):
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
    print("taking image")
    max_ret = MAX_RETRIES
    update.message.reply_text("Aspetta qualche secondo...")



    #try to read the image
    ret, img = CAM.read()

    #while the reading is unsuccesfull
    while not ret:
        #read again and sleep
        ret, img = CAM.read()
        sleep(1)
        max_ret -= 1
        #if max retries is exceeded exit and release the stream
        if max_ret == 0:
            update.message.reply_text("Ci sono stati dei problemi tecnici 1...riprova")
            cv2.VideoCapture(0).release()
            return


    #try to save the image
    ret = cv2.imwrite(image, img)
    max_ret = MAX_RETRIES

    while not ret:
        ret = cv2.imwrite(image, img)
        sleep(1)
        max_ret -= 1
        #if max retries is exceeded exit and release the stream

        if max_ret == 0:
            update.message.reply_text("Ci sono stati dei problemi tecnici 2...riprova")
            cv2.VideoCapture(0).release()
            return


    cv2.VideoCapture(0).release()


    if ret:
        sleep(2)
        print("image taken")
        update.message.reply_text("Invio immagine")
        with open(image, "rb") as file:
            bot.sendPhoto(update.message.from_user.id, file)
        os.remove(image)
