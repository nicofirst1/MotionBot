from telegram.ext import (
    ConversationHandler, CommandHandler, MessageHandler, Filters, CallbackQueryHandler)

from handlers import start, get_psw, annulla, get_camshot, stream, disp, updater, notification, face_detection, \
    flag_setting_main, flag_setting_callback

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
    disp.add_handler(CommandHandler("flags",flag_setting_main))
    disp.add_handler(CallbackQueryHandler(flag_setting_callback, pattern="/flag"))

    print("Polling...")
    updater.start_polling()
