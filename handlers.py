from time import sleep
from telegram.ext import ConversationHandler
from subprocess import call
import cv2
import os

MAX_RETRIES=8
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

    image="image.png"
    print("taking image")
    max_ret=MAX_RETRIES
    update.message.reply_text("Aspetta qualche secondo...")
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()

    while not ret:
        ret, img = cap.read()
        sleep(1)
        max_ret-=1
        if max_ret==0:
            update.message.reply_text("Ci sono stati dei problemi tecnici 1...riprova")
            break

    if not ret:
        return

    ret= cv2.imwrite(image, img)
    max_ret=MAX_RETRIES

    while not ret:
        ret = cv2.imwrite(image, img)
        sleep(1)
        max_ret -= 1
        if max_ret == 0:
            update.message.reply_text("Ci sono stati dei problemi tecnici 2..riprova")
            break
    if not ret:
        return

    cv2.VideoCapture(0).release()
    sleep(2)
    print("image taken")

    if ret:
        update.message.reply_text("Invio immagine")
        with open(image,"rb") as file:
            bot.sendPhoto(update.message.from_user.id, file)
        os.remove(image)

