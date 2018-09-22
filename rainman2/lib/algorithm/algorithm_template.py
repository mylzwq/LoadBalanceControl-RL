#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base model definition for creating any algorithm modules
"""

import logging
from loadbalanceRL.utils import exceptions


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Tuesday, February 20th 2018, 1:37:55 pm'

LOGGER = logging.getLogger(__name__)


class Base:
    # pylint: disable=too-few-public-methods
    """
    Parent class for all the algorithm classes
    """
    def __new__(cls, alg_config, env, *args):
        """
        Method allows to create new algorithm class without having them to
        call Base class everytime.
        """
        base = super(Base, cls).__new__(cls)
        base.env = env
        base.episodes = alg_config['EPISODES']
        base.alpha = alg_config['ALPHA']
        base.gamma = alg_config['GAMMA']
        base.epsilon = alg_config['EPSILON']
        base.epsilon_decay = alg_config['EPSILON_DECAY']
        base.epsilon_min = alg_config['EPSILON_MIN']
        base.verbose = alg_config['VERBOSE']
        return base

    # Set respective agents per algorithm sub class
    agent = None

    # Override these private methods per algorithm basis
    def _execute(self):
        # pylint: disable=no-self-use
        """
        Private method containing implementation details for respective alg.

        Every Algorithm class must implement this method else the program
        will raise exception and stop.
        """

        raise exceptions.AlgorithmMethodNotImplemented(
            "_execute method is not implemented! Please check your algorithm"
        )

    def execute(self):
        """
        Public method to execute requested algorithm on the given
        environment

        Returns:
            output: (tuple)

        Raises:
            AlgorithmMethodNotImplemented
        """

        return self._execute()
