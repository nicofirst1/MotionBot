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

    update.message.reply_text("Aspetta un secondo...")
    cap = cv2.CaptureFromCAM(0)
    img = cv2.QueryFrame(cap)
    cv2.imwrite('image.png', img)
    cv2.VideoCapture(0).release()
    sleep(2)
    print("image taken")

    if "image.png" in os.listdir("."):
        print("image found")
        with open("image.png","rb") as file:
            bot.sendPhoto(update.message.from_id,file)