import sys
import os
import logging


def simple_log(log, output_base, prefix, log_level="NOTSET", file_handler_mode="a"):
    log_simple = log
    for handler in list(log_simple.handlers):
        log_simple.removeHandler(handler)
    if log_level == "INFO":
        log_simple.setLevel(logging.INFO)
    else:
        log_simple.setLevel(logging.NOTSET)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    if log_level == "INFO":
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.NOTSET)
    logfile = logging.FileHandler(os.path.join(output_base, prefix + '.log.txt'), mode=file_handler_mode)
    logfile.setFormatter(logging.Formatter('%(message)s'))
    if log_level == "INFO":
        logfile.setLevel(logging.INFO)
    else:
        logfile.setLevel(logging.NOTSET)
    log_simple.addHandler(console)
    log_simple.addHandler(logfile)
    return log_simple


def timed_log(log, output_base, prefix, log_level="NOTSET", file_handler_mode="a"):
    log_timed = log
    for handler in list(log_timed.handlers):
        log_timed.removeHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    if log_level == "INFO":
        console.setLevel(logging.INFO)
    else:
        console.setLevel(logging.NOTSET)
    logfile = logging.FileHandler(os.path.join(output_base, prefix + '.log.txt'), mode=file_handler_mode)
    logfile.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
    if log_level == "INFO":
        logfile.setLevel(logging.INFO)
    else:
        logfile.setLevel(logging.NOTSET)
    log_timed.addHandler(console)
    log_timed.addHandler(logfile)
    return log_timed
