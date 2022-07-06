import logging
import os

def set_logging_defaults(time_data,logdir):
    # set basic configuration for logging
    # logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
    logging.basicConfig(format="[%(asctime)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir,time_data, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)],
                        datefmt = '%m-%d %I:%M:%S')

    # log cmdline argumetns
    logger = logging.getLogger('main')
    logger.info(' '.join(os.sys.argv))


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs