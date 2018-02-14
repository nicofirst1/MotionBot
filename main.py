from telegram.ext import (
    ConversationHandler, CommandHandler, MessageHandler, Filters, CallbackQueryHandler)
import logging
from handlers import start, get_psw, annulla, get_camshot, stream, disp, updater, \
    flag_setting_main, flag_setting_callback, reset_ground, stop_execution, send_log



logger = logging.getLogger('motionlog')
hdlr = logging.FileHandler('motion.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)



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
    disp.add_handler(CommandHandler("resetg",reset_ground))
    disp.add_handler(CommandHandler("stop",stop_execution))
    disp.add_handler(CommandHandler("log",send_log))
    disp.add_handler(CallbackQueryHandler(flag_setting_callback, pattern="/flag"))

    print("Polling...")
    logger.info("Start polling")
    updater.start_polling()
