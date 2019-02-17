import shutil
from os import curdir
import os
from os.path import join

class Path:

    curdir=os.getcwd()
    curdir=curdir.split("MotionBot")[0]
    curdir=join(curdir,"MotionBot")

    RESOURCES_DIR = join(curdir, "Resources")
    LOGGER_DIR = join(RESOURCES_DIR, "Loggers")
    WEIGHTS_DIR=join(RESOURCES_DIR,"Weights")
    CUSTOM_DARK_DIR=join(RESOURCES_DIR,"Custom_darknet")
    ANALISYS_DIR=join(RESOURCES_DIR,"Analisys")
    recognizer=join(RESOURCES_DIR,"recognizer.yaml")

    model=join(RESOURCES_DIR,"model.pkl")
    time_profiler=join(RESOURCES_DIR,"time_profiler")
    token_psw=join(RESOURCES_DIR,"token_psw.txt")
    ids=join(RESOURCES_DIR,"ids")
    encodings="encodings.pkl"

    DARKNET_DIR=join(curdir,"src/darknet")



    FACES_DIR=join(curdir,"Faces")
    UNK_DIR=join(FACES_DIR,"Unknown")

    def __init__(self):
        self.initialize_dirs()

    def initialize_dirs(self):
        """
        Initialize all the directories  listed above
        :return:
        """
        variables = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for var in variables:
            if var.endswith('DIR'):
                path = getattr(self, var)
                if not os.path.exists(path):
                    os.makedirs(path)

    def empty_dirs(self, to_empty):
        """
        Empty all the dirs in to_empty
        :return:
        """

        for folder in to_empty:
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)

    @staticmethod
    def join(path1,path2):
        return join(path1,path2)

