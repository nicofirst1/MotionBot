import logging
from Path import Path as pt


def initialize_logger(base_name):
    """
    Initialize a logger with a given base name
    :param base_name: (str) the base name to give to the logger
    :return: logger
    """
    logger = logging.getLogger(f"{base_name}")
    hdlr = logging.FileHandler(f'{pt.LOGGER_DIR}/{base_name}.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s : %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger



