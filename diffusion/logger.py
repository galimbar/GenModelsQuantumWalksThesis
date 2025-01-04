import logging

def logger(logging_level = logging.DEBUG):
    logging.basicConfig(level=logging_level,
                           format='%(asctime)s %(levelname)s %(message)s',
                           datefmt="%Y-%m-%d %H:%M:%S",
                           filename="Diff_log.log")
    logging.debug("\n\n##########################################################################\n\nlogging started\n\n##########################################################################\n\n")
