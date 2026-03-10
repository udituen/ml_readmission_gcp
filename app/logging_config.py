import logging
import json
import time

def get_logger():
    logger = logging.getLogger("ml-api")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        '{"timestamp":"%(asctime)s","level":"%(levelname)s","message":%(message)s}'
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger