# coding=utf-8
from datetime import datetime
from functools import wraps
import cProfile, pstats, io
import datetime
MAX_RETRIES = 8

COMMANDS="""
photo - send camshot
start - start bot
video - send video 
flags - set flags
resetg - reset the groung image
stop - stop surveillance execution
logsend - send the logger file
logdel - delete the log file
bkground - send the background image
classify - classify the person face
help - send the help text
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
    ids_path="Resources/ids"

    with open(ids_path,"w+") as file:
        for elem in ids:
            file.write(str(elem[0])+","+str(elem[1]))


def read_ids():
    ids_path="Resources/ids"


    with open(ids_path, "r+") as file:
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
    """Function to read token and password from file"""
    file_name="Resources/token_psw.txt"

    #read file
    with open(file_name, "rb") as file:
        lines=file.readlines()

    #take token and password
    token=lines[0].split(b"=")[1].strip(b"\n")
    psw=lines[1].split(b"=")[1].strip(b"\n")

    #return converted to string
    return token.decode("utf-8") ,psw.decode("utf-8")



def time_profiler():

    def real_decorator(function):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            # take the time
            start = datetime.datetime.now()
            max_seconds=7

            function(*args, **kwargs)

            end = datetime.datetime.now()

            if (end-start).seconds<max_seconds: return

            pr.disable()
            s = io.StringIO()
            sortby = 'tottime'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            to_print=""
            to_print += "\n\n=====================TIMER PROFILER START========================\n"
            to_print += "\n"+str(function)+  "\n"
            to_print +=str(s.getvalue())
            to_print +="\n=====================TIMER PROFILER END========================\n\n"



            try:
                with open("Resources/time_profiler", "a+") as file:
                    file.write(to_print)
            except FileNotFoundError:
                print("Time profiler file not found")
                pass


        return wrapper
    return real_decorator



def memory_profiler():

    def real_decorator(function):
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()

            function(*args, **kwargs)

            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())


        return wrapper
    return real_decorator