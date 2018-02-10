from telegram.ext import (
     ConversationHandler, CommandHandler, MessageHandler, Filters)

from handlers import start, get_psw, annulla, get_camshot, stream, disp, updater, notification, face_detection

if __name__ == "__main__":


    conversation = ConversationHandler(
                    [CommandHandler("start", start)],
                    states={
                        1: [MessageHandler(Filters.text, get_psw)],

                    },
                    fallbacks=[CommandHandler('Fine', annulla)]
                )

    disp.add_handler(conversation)

    disp.add_handler(CommandHandler("photo",get_camshot))
    disp.add_handler(CommandHandler("video",stream,pass_args=True))
    disp.add_handler(CommandHandler("notification",notification,pass_args=True))
    disp.add_handler(CommandHandler("facedetection",face_detection,pass_args=True))

    print("Polling...")
    updater.start_polling()
