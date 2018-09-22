#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base model definition for creating any environment models
"""

import logging
from loadbalanceRL.utils import exceptions

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, February 21st 2018, 4:40:29 pm'


LOGGER = logging.getLogger(__name__)


class Base:
    """
    Parent class for all the envrionment classes
    """
    def __new__(cls, env_config, *args):
        """
        Method allows to create new environment class without having them to
        call Base class everytime.
        """
        base = super(Base, cls).__new__(cls)
        base.env_config = env_config
        base.env_name = env_config['NAME']
        base.verbose = env_config['VERBOSE']
        return base

    initial_state = None

    # Override these private methods per environment basis
    @property
    def _actions(self):
        """
        Private method that implements actions available for the environment
        """
        raise exceptions.EnvironmentMethodNotImplemented(
            "_actions is not implemented for this environment!"
        )

    @property
    def _state_dim(self):
        """
        Private method to retrieve shape of the state declared by the
        environemnt.
        """
        raise exceptions.EnvironmentMethodNotImplemented(
            "_state_dim is not implemented for this environment!"
        )

    def _reset(self):
        """
        Private method to reset the environment's state
        """
        raise exceptions.EnvironmentMethodNotImplemented(
            "_reset() is not implemented for this environment!"
        )

    def _reset_after_move(self):
        """
        Private method to reset the environment's state
        """
        raise exceptions.EnvironmentMethodNotImplemented(
            "_reset_after_move() is not implemented for this environment!"
        )

    def _step(self, state, action, *args):
        """
        Private method that simulates a time step for the environment
        """
        raise exceptions.EnvironmentMethodNotImplemented(
            "_step() is not implemented for this environment!"
        )

    @property
    def actions(self):
        """
        Method to return all the actions defined by the environment
        """
        return self._actions

    @property
    def n_actions(self):
        """
        Method to return number of actions possible
        """
        return len(self.actions)

    @property
    def state_dim(self):
        """
        Method to return dimension of the environment's state
        """
        return self._state_dim

    def reset(self):
        """
        Method to reset environment's state
        """
        return self._reset()

    def reset_after_move(self):
        """
        Method to reset environment's state
        """
        return self._reset_after_move()

    def step(self, state, action, *args):
        """
        Method to implement how envrionment simulates time step
        """
        return self._step(state, action, *args)
