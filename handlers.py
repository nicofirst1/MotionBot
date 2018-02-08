from time import sleep

from telegram.ext import ConversationHandler
from subprocess import call
import cv2
import os

psw="SuperMegaFamBrand123!"



def start(bot,update):
    update.message.reply_text("Benvenuto...per iniziare inserisci la password")
    return 1

def get_psw(bot, update):

    user_psw=update.message.text

    if not user_psw==psw:
        update.message.reply_text("Password incorretta")

    else:
        update.message.reply_text("Password corretta")


def annulla(bot,update):
    update.message.reply_text("annullo")
    return ConversationHandler.END


def get_camshot(bot, update):
    print("taking image")
    max_ret=4
    update.message.reply_text("Aspetta un secondo...")
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()

    while not ret:
        ret, img = cap.read()
        sleep(1)
        max_ret-=1
        if max_ret==0:
            update.message.reply_text("Ci sono stati dei problemi tecnici 1")
            break

    ret= cv2.imwrite('image.png', img)
    max_ret=4

    while not ret:
        ret = cv2.imwrite('image.png', img)
        sleep(1)
        max_ret -= 1
        if max_ret == 0:
            update.message.reply_text("Ci sono stati dei problemi tecnici 2")
            break

    cv2.VideoCapture(0).release()
    sleep(2)
    print("image taken")

    if ret:
        update.message.reply_text("Invio immagine")
        with open("image.png","rb") as file:
        bot.sendPhoto(update.message.from_id,file)