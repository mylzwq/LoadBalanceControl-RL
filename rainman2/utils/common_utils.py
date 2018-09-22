#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""

General Purporse Convenience Tools

"""

import os
import time
import logging
import yaml
import simplejson as json
from loadbalanceRL.utils import exceptions


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, February 14th 2018, 11:36:05 am'


def timeit(function):
    """
    Decorator to measure time take to execute a function
    """
    def wrapper(*args):
        """
        Wrapper definition for the function
        """
        start_time = time.process_time()
        output = function(*args)
        print(
            "Module: {} took: {}s".format(
                function.__name__,
                time.process_time() - start_time))
        return output
    return wrapper


def load_yaml(file_to_open):
    """
    Helper function to load a yaml file.

    Args:
        file_to_open (str):
            File name

    Returns:
        content (dict):
            content of the file

    Raises:
        FileOpenError
    """

    logger = logging.getLogger(__name__)

    if not os.path.isfile(file_to_open):
        error = "Couldn't find file: %s", file_to_open
        logger.exception(error)
        raise exceptions.FileOpenError(error)
    with open(file_to_open, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


def load_json(file_to_open):
    """
    Helper function to load a Json file.

    Args:
        file_to_open (str):
            File name

    Returns:
        content (dict):
            content of the file

    Raises:
        FileOpenError
    """
    logger = logging.getLogger(__name__)

    if not os.path.isfile(file_to_open):
        error = "Couldn't find file: %s", file_to_open
        logger.exception(error)
        raise exceptions.FileOpenError(error)
    with open(file_to_open, 'r') as json_file:
        return json.load(json_file)
