#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Logging support

"""
import logging
import logging.config
import os

import coloredlogs
import loadbalanceRL.constants as CONSTANTS
from loadbalanceRL.utils import common_utils, exceptions

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, February 14th 2018, 11:05:29 am'


def update_config(config_file, new_log_file):
    """
    Helper function to update log config json
    """
    config_file['handlers']['file_handler']['filename'] = new_log_file
    config_file['handlers']['info_handler']['filename'] = new_log_file
    return config_file


def setup_logging(log_config=CONSTANTS.LOG_CONFIG_FILE,
                  log_level=logging.info,
                  log_file=CONSTANTS.LOG_FILE):
    """
    Function to setup logging
    """
    if not os.path.exists(CONSTANTS.LOG_DIR):
        os.makedirs(CONSTANTS.LOG_DIR)

    try:
        config_file = common_utils.load_json(log_config)
    except exceptions.FileOpenError:
        print("Failed to load configuration file. Using default configs")
        logging.basicConfig(level=log_level)
        coloredlogs.install(level=log_level, milliseconds=True)
    else:
        updated_log_config = update_config(config_file, log_file)
        logging.config.dictConfig(updated_log_config)
        coloredlogs.install(milliseconds=True)
        print("Rainman2's logging has been configured!")
