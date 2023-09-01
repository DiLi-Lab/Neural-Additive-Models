import logging
import os
from datetime import datetime


class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG


def get_logger(path: str):
    if not os.path.isdir(path + '/hparams_logs'):
        os.mkdir(path + '/hparams_logs')

    # configuration for log file
    log = logging.getLogger('rf-baseline-grid-search')
    log.setLevel(logging.DEBUG)

    # configure separate logging handlers for info and debug to separate files
    info_log = logging.FileHandler(
        f'{path}/hparams_logs/grid_search_rf_baseline_info_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    info_log.setLevel(logging.INFO)
    info_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))

    debug_log = logging.FileHandler(
        f'{path}/hparams_logs/grid_search_rf_baseline_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    debug_log.setLevel(logging.DEBUG)
    debug_log.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    # configure filter to ignore all messages higher than debug for the debug file
    debug_log.addFilter(DebugFilter())

    log.addHandler(info_log)
    log.addHandler(debug_log)

    # log info logger filename to debug handler
    log.debug(f'Info logger filename: {info_log.baseFilename}')

    return log


def create_results_folders(script: str, name: str = 'results'):
    today = datetime.now().strftime('%Y-%m-%d')
    full_name = f'{name}/{script}_{today}'

    # create results folder if it does not exist
    if not os.path.isdir(full_name):
        os.makedirs(full_name)

    return full_name
