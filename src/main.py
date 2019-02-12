

from telegram.ext import ConversationHandler, CommandHandler, MessageHandler, Filters, CallbackQueryHandler

from src.Utils.logger import initialize_logger
from src.handlers import start, annulla, get_camshot, stream, disp, updater, \
    flag_setting_main, flag_setting_callback, reset_ground, stop_execution, send_log, send_ground, get_psw, delete_log, \
    help_bot, predict_face

#Implementing logger
loggers=["cam_shotter","cam_movement","telegram_handler","dark_net","main_class"]

for log in loggers:
    logger=initialize_logger(log)


if __name__ == "__main__":

    #Adding converation handler
    conversation = ConversationHandler(
                    [CommandHandler("start", start)],
                    states={
                        1: [MessageHandler(Filters.text, get_psw)],

                    },
                    fallbacks=[CommandHandler('Fine', annulla)]
                )

    disp.add_handler(conversation)

    #Adding Command handler
    disp.add_handler(CommandHandler("photo",get_camshot))
    disp.add_handler(CommandHandler("video",stream,pass_args=True))
    disp.add_handler(CommandHandler("flags",flag_setting_main))
    disp.add_handler(CommandHandler("resetg",reset_ground))
    disp.add_handler(CommandHandler("stop",stop_execution))
    disp.add_handler(CommandHandler("logsend",send_log))
    disp.add_handler(CommandHandler("bkground",send_ground))
    disp.add_handler(CommandHandler("logdel",delete_log))
    disp.add_handler(CommandHandler("help",help_bot))
    #Adding CallcbackQuery
    disp.add_handler(CallbackQueryHandler(flag_setting_callback, pattern="/flag"))
    #adding message handler
    disp.add_handler(MessageHandler(Filters.photo,predict_face))


    print("Polling...")
    logger.info("Start polling")
    updater.start_polling()
