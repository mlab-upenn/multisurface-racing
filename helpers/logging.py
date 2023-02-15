import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def create_logger(name, log_level):
    # create logger with name
    logger = logging.getLogger(name)
    logger.level = log_level

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Add formatting to the console handler
    ch.setFormatter(CustomFormatter())

    # add the console handler to the logger
    logger.addHandler(ch)
    
    return logger