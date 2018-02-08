from telegram.ext import (
    Updater, ConversationHandler, CommandHandler, MessageHandler, Filters)

import os

from handlers import start, get_psw, annulla, get_camshot

PORT = int(os.environ.get('PORT', '5000'))

token="""545431258:AAHEocYDtLOQdZDCww6tQFSfq3p-xmWeyE8"""


updater = Updater(token)
disp = updater.dispatcher





conversation = ConversationHandler(
                [CommandHandler("start", start)],
                states={
                    1: [MessageHandler(Filters.text, get_psw)],

                },
                fallbacks=[CommandHandler('Fine', annulla)]
            )

disp.add_handler(conversation)

disp.add_handler(CommandHandler("photo",get_camshot))


print("Polling...")
updater.start_polling()
