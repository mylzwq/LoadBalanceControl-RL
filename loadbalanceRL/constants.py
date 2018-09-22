/*
 * @Author: Ari Saha, Mingyang Liu
 * @Date: 2018-09-19 00:27:56
 * @LastEditors: Mingyang Liu
 * @LastEditTime: 2018-09-19 00:28:34
 * @Description: 
 * @Email: liux3941@umn.edu
 */
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rainman2 constants' declaration
These are factory settings
"""

import os
from collections import OrderedDict

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu (liux3941@umn.edu)'

# Software settings
APP_DIR = os.path.abspath(os.path.dirname(__file__))
ETC_DIR = os.path.join(APP_DIR, 'etc')
LIB_DIR = os.path.join(APP_DIR, 'lib')
API_DIR = os.path.join(APP_DIR, 'api')
LOG_DIR = os.path.join(APP_DIR, 'log')
LOG_FILE = os.path.join(LOG_DIR, 'loadbalanceRL.log')
LOG_CONFIG_FILE = os.path.join(ETC_DIR, 'logging.json')
CONFIG_OVERRIDES = os.path.join(ETC_DIR, 'overrides.yml')
VERBOSE = False

# Algorithm settings
ALGORITHM_CONFIG = OrderedDict(
    EPISODES=1,
    ALPHA=0.5,
    GAMMA=0.9,
    EPSILON=0.1,
    EPSILON_DECAY=0.999,
    EPSILON_MIN=0.01,
    VERBOSE=VERBOSE,
    LEARNING_RATE=0.005,
    L1_HIDDEN_UNITS=13,
    L2_HIDDEN_UNITS=13,
    L1_ACTIVATION='relu',
    L2_ACTIVATION='relu',
    LOSS_FUNCTION='mean_squared_error',
    OPTIMIZER='Adam',
    REPLACE_TARGET_ITER=20,
    MEMORY_SIZE=2000,
    BATCH_SIZE=16,
)

# Environment settings

# Default environment is set to Cellular Dev
DEFAULT_ENVIRONMENT = 'Cellular'
DEFAULT_CELLULAR_TYPE = 'Dev'
DEFAULT_BACKBONE_TYPE = 'Dev'

# 1) For Cellular environment
CELLULAR_MODEL_CONFIG = OrderedDict(
    NAME='Cellular',
    TYPE=DEFAULT_CELLULAR_TYPE,
    SERVER='0.0.0.0',
    SERVER_PORT='8000',
    VERBOSE=VERBOSE,
)

# For testing with sample environment
SAMPLE_ENV_CONFIG = OrderedDict(
    NAME='General',
    VERBOSE=True
)

# 2) For Backbone (e.g. core network) environment
BACKBONE_MODEL_CONFIG = OrderedDict(
    TYPE=DEFAULT_BACKBONE_TYPE,
)

ENVIRONMENT_DICT = {
    'Cellular': CELLULAR_MODEL_CONFIG,
    'Backbone': BACKBONE_MODEL_CONFIG,
}
