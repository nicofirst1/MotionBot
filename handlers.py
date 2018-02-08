from telegram.ext import ConversationHandler
from subprocess import call
import cv2


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
    cap = cv2.VideoCapture(0)

    ret, img = cap.read()
    cv2.imshow("input", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()
    print("image taken")
