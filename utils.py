# coding=utf-8

from functools import wraps
#from inspect import signature
from time import sleep
import cv2
from datetime import datetime

CAM=cv2.VideoCapture(0)
MAX_RETRIES = 8

COMMANDS="""
photo - invia una foto
start - inizzializza il bot
"""

def is_enabled(id):
    IDS=read_ids()
    for elem in IDS:
        if elem[0]==str(id):
            if not int(elem[1]):
                return -1
            else:
                return 1
    return 0


def add_id(user_id,enabled):
    ids = read_ids()
    ids.append((user_id,enabled))
    write_ids(ids)
    return ids


def write_ids(ids):

    with open("ids","w+") as file:
        for elem in ids:
            file.write(str(elem[0])+","+str(elem[1]))


def read_ids():
    with open("ids", "r+") as file:
        ids = file.readlines()

    ids=[(elem.split(",")[0],elem.split(",")[1].strip("\n")) for elem in ids]
    return ids


def elegible_user(func):
    """questa funzione ha il compito di verificare se l'id utente è abilitato a chiamare il comando
    il suo utilizzo è il seguente:
    data la funzione command che deve essere wrappata, si può creare una nuova funzione elegible_user(command) """

    @wraps(func)
    def check_if_user_can_interact(bot, update, *args, **kwargs):
        """Questa funzione ritorna true se l'user puo interagire, altrimenti false
        inoltre in caso di false (user non presente nel db inizia il procedimento di richiesta d'accesso"""

        user_id = update.message.from_user.id

        value=is_enabled(user_id)

        #user non presente nel file
        if value==0:
            # se il messaggio è stato mandato in privata allora devo chiedere l'accesso
            if 'private' in update.message.chat.type:
                update.message.reply_text("Usa /start per accedere alle funzionalità del bot")
                return

        #user bannato
        if value==-1:
            update.message.reply_text("Non sei abilitato ad usare il bot")
            return

        else:
            # sig = signature(func)
            # if len(sig.parameters) > 1:
            #     return func(bot, update, *args, **kwargs)
            # else:
            #     return func(*args, **kwargs)
            return func(bot, update, *args, **kwargs)

    return check_if_user_can_interact


def capture_image(image_name):
    max_ret = MAX_RETRIES

    # try to read the image
    ret, img = CAM.read()

    # while the reading is unsuccesfull
    while not ret:
        # read again and sleep
        ret, img = CAM.read()
        sleep(1)
        max_ret -= 1
        if not ret:
            cv2.VideoCapture(0).release()
            CAM.cv2.VideoCapture(0)
        # if max retries is exceeded exit and release the stream
        if max_ret == 0:
            cv2.VideoCapture(0).release()
            return False

    # try to save the image
    ret = cv2.imwrite(image_name, img)
    max_ret = MAX_RETRIES

    while not ret:
        ret = cv2.imwrite(image_name, img)
        sleep(1)
        max_ret -= 1
        # if max retries is exceeded exit and release the stream

        if max_ret == 0:
            cv2.VideoCapture(0).release()
            return False

    cv2.VideoCapture(0).release()
    return True



def capture_video(video_name,seconds):
    frame_width = 1024
    frame_height = 720
    print(frame_height,frame_width)
    fps=10
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    start=datetime.now()
    end=datetime.now()
    while (True):
        ret, frame = CAM.read()

        if ret == True:

            # Write the frame into the file 'output.avi'
            out.write(frame)

        # Break the loop
        else:
            pass

        if (end - start).seconds>=seconds:
            break

        end=datetime.now()

            # When everything done, release the video capture and video write objects
    CAM.release()
    out.release()
