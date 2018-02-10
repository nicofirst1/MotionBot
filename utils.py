# coding=utf-8

from functools import wraps

MAX_RETRIES = 8

COMMANDS="""
photo - send camshot
start - start bot
video - send video 
notification - ON/OFF
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


def read_token_psw():
    file_name="token_psw.txt"

    with open(file_name, "rb") as file:
        lines=file.readlines()

    token=lines[0].split(b":")[1]
    psw=lines[1].split(b":")[0]
    print(token,psw)

    return token,psw
