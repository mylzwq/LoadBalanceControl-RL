#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rainman2 settings
Use this to load runtime config
"""

import logging
import loadbalanceRL.constants as CONSTANTS
from loadbalanceRL.utils import common_utils
from loadbalanceRL.utils import exceptions

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


class Setting:
    """
    Rainman2's default settings
    """

    def __init__(self, runtime_config):
        """
        Initialize settings
        """
        self.logger = logging.getLogger(__name__)

        # Logging setup
        self.log_dir = CONSTANTS.LOG_DIR

        # algorithm settings
        self.logger.info("Loading default config values")
        self.algorithm_config = CONSTANTS.ALGORITHM_CONFIG

        # environment settings
        # by default Rainman2 loads cellular environmemt
        def update_env(env):
            """
            Helper to assign correct environment config
            """
            if env not in CONSTANTS.ENVIRONMENT_DICT:
                raise exceptions.EnvironmentNotImplemented(
                    "Env: {} doesn't exist!".format(env)
                )
            config = CONSTANTS.ENVIRONMENT_DICT[env]
            return config

        self.update_env = update_env

        self.environment_config = self.update_env(
            CONSTANTS.DEFAULT_ENVIRONMENT
        )

        # Update config with overrides file values
        if runtime_config:
            self.logger.info("Updating default config values")
            self.algorithm_config['EPISODES'] = runtime_config['episodes']
            self.algorithm_config['ALPHA'] = runtime_config['alpha']
            self.algorithm_config['GAMMA'] = runtime_config['gamma']
            self.algorithm_config['EPSILON'] = runtime_config['epsilon']
            self.algorithm_config['EPSILON_DECAY'] =\
                runtime_config['epsilon_decay']
            self.algorithm_config['EPSILON_MIN'] =\
                runtime_config['epsilon_min']
            self.algorithm_config['VERBOSE'] = runtime_config['verbose']

            # Neural net config
            self.algorithm_config['L1_HIDDEN_UNITS'] =\
                runtime_config['l1_hidden_units']
            self.algorithm_config['L2_HIDDEN_UNITS'] =\
                runtime_config['l2_hidden_units']
            self.algorithm_config['L1_ACTIVATION'] =\
                runtime_config['l1_activation']
            self.algorithm_config['L2_ACTIVATION'] =\
                runtime_config['l2_activation']
            self.algorithm_config['LOSS_FUNCTION'] =\
                runtime_config['loss_function']
            self.algorithm_config['OPTIMIZER'] =\
                runtime_config['optimizer']
            self.algorithm_config['LEARNING_RATE'] =\
                  runtime_config['learning_rate']
            self.algorithm_config['REPLACE_TARGET_ITER'] =\
                  runtime_config['replace_target_iter']
            self.algorithm_config['MEMORY_SIZE'] =\
                  runtime_config['memory_size']
            self.algorithm_config['BATCH_SIZE'] =\
                  runtime_config['batch_size']

            self.environment_config = self._update_environment_config(
               runtime_config['environment']
            )
            if runtime_config['environment'] == 'Cellular':
                self.environment_config['TYPE'] = runtime_config['env_type']
                self.environment_config['SERVER'] =\
                    runtime_config['server']
                self.environment_config['SERVER_PORT'] =\
                    runtime_config['server_port']
                self.environment_config['VERBOSE'] =\
                    runtime_config['verbose']
            elif runtime_config['environment'] == 'Backbone':
                self.environment_config['TYPE'] = runtime_config['env_type']

    def _update_environment_config(self, env):
        """
        Helper to return environment config based on env name
        """
        return CONSTANTS.ENVIRONMENT_DICT[env]


def update_runtime_config():
    """
    Function to update run time configurations
    """
    logger = logging.getLogger(__name__)
    try:
        runtime_config = common_utils.load_yaml(CONSTANTS.CONFIG_OVERRIDES)
    except exceptions.FileOpenError as error:
        logger.debug("Error: %s", error)
        return None
    else:
        return runtime_config


SETTINGS = Setting(update_runtime_config())
