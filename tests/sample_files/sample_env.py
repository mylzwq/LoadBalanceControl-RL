#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Sample environment for testing """

from collections import OrderedDict
from collections import namedtuple
from loadbalanceRL.lib.environment import environment_template

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


ACTIONS = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT'
}

STATE_ATTRIBUTES = OrderedDict(
    attr1=None,
    attr2=None,
    attr3=None,
)

STATE = namedtuple(
    'STATE',
    STATE_ATTRIBUTES.keys())

STATES_LIST = [STATE('initial', 1, 1),
               STATE('next', 0.5, 0.5),
               STATE('last', 0.2, 0.9)]


class SampleGeneralEnv(environment_template.Base):
    """
    Sample environment
    """
    def __init__(self, env_config):
        """
        Initialize sample env
        """
        self.env_config = env_config
        self.states = iter(STATES_LIST)

    def get_next_state(self):
        """
        Fetches next state
        """
        return next(self.states, None)

    @property
    def _actions(self):
        """
        Sample actions allowed in the env
        """
        return ACTIONS

    @property
    def _state_dim(self):
        """
        Sample state dim
        """
        return len(STATE_ATTRIBUTES)

    def _reset(self):
        """
        Sample action to reset the env
        """
        self.states = iter(STATES_LIST)
        return self.get_next_state()

    def _step(self, state, action):
        """
        Sample action to take a step
        """
        next_state = self.get_next_state()
        stop = False
        reward = 1
        if next_state.attr1 == 'last':
            stop = True
        return next_state, reward, stop
