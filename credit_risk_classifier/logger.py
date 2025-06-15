import logging
import os

def get_logger(name=__name__, log_file="ml_pipeline.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

 
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(os.getcwd(), log_file), mode="a")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

