#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Rainman2

"""

from loadbalanceRL.utils import logging_utils
from loadbalanceRL.lib import interface
from loadbalanceRL.settings import SETTINGS


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


# version
__version__ = '1.0'

# logging setup
logging_utils.setup_logging()

RAINMAN3 = interface.Rainman2(SETTINGS)
