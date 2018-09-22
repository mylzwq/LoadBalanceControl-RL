#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Base model definition for creating any agents
"""

import logging
from loadbalanceRL.utils import exceptions

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, February 21st 2018, 1:40:20 pm'

LOGGER = logging.getLogger(__name__)


class Base:
    """
    Parent class for all agents
    """
    def __new__(cls, alg_config, agent_config):
        """
        Method allows to create new agent class without having to call Base
        class everytime.
        """
        base = super(Base, cls).__new__(cls)
        base.alg_config = alg_config
        base.agent_config = agent_config
        base.episodes = alg_config['EPISODES']
        base.alpha = alg_config['ALPHA']
        base.gamma = alg_config['GAMMA']
        base.epsilon = alg_config['EPSILON']
        base.epsilon_decay = alg_config['EPSILON_DECAY']
        base.epsilon_min = alg_config['EPSILON_MIN']
        base.verbose = alg_config['VERBOSE']

        # set actions for each agent. Actions must be defined by environment
        base.n_actions = agent_config.n_actions
        # set state_dim for certain agents.
        # state dim must be defined by environment
        base.state_dim = agent_config.state_dim

        return base

    model = None

    # Override these private methods for each agent
    def _build_model(self):
        """
        Private method that implements the logic to compute Q(s, a)
        """
        raise exceptions.AgentMethodNotImplemented(
            "_build_model is not implemented for this agnet!"
        )

    def _take_action(self, state, *args):
        """
        Private method that implements how an agent takes actions given
        a state.
        """
        raise exceptions.AgentMethodNotImplemented(
            "_take_action is not implemented for this agent!"
        )

    def _learn(self, state, action, reward, next_state, *args):
        """
        Private method that implements how an agent learns given current
        state,current action, reward and next_state.
        """
        raise exceptions.AgentMethodNotImplemented(
            "_learn method it not implemented for this agent!"
        )

    def take_action(self, state, *args):
        """
        Public method that implements how an agent takes actions given a state

        Args:
            state: (tuple)
                Current state of the envrionment

        Returns:
            action: (int)
                next action to take based on current state
        """
        return self._take_action(state, *args)

    def learn(self, state, action, reward, next_state, *args):
        """
        Public method that implements how an agent learns given state, action,
        reward and next_state

        Args:
            state: (tuple)
                Current state of the envrionment
            action: (int)
                Current action determined by the policy
            reward: (float)
                reward based on the action taken by the agent
            next_state: (tuple)
                Next state based on the action taken by the agent

        """
        return self._learn(state, action, reward, next_state, *args)
