#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Controller to dispatch respective Qlearning instance """

import logging
from collections import namedtuple
from loadbalanceRL.utils import exceptions
from loadbalanceRL.lib.algorithm import algorithm_template
from loadbalanceRL.lib.algorithm.Qlearning.agents import regression2
from loadbalanceRL.lib.algorithm.Qlearning.agents import neural_nets2
from loadbalanceRL.lib.algorithm.Qlearning.agents import DeepQN
from loadbalanceRL.lib.algorithm.Qlearning.agents import tabular_q_learning
from loadbalanceRL.lib.algorithm.Qlearning.general import QlearningForGeneral
from loadbalanceRL.lib.algorithm.Qlearning.cellular import QlearningForCellular

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, April 4th 2018, 10:20:27 am'

SUPPORTED_ENV = {
    'General': QlearningForGeneral,
    'Cellular': QlearningForCellular,
}

QLEARNING_AGENTS = {
    'General': {
        'Naive': tabular_q_learning.QNaiveAgent,
        'LinearRegression': regression2.QLinearRegressionAgent,
        'NN': neural_nets2.QNNAgent,
    },

    'Cellular': {
        'Naive': tabular_q_learning.QCellularAgent,
        'LinearRegression': regression2.QCellularLinearRegressionAgent,
        'NN': neural_nets2.QCellularNNAgent,
        'DQN':DeepQN.DQNCellularAgent,
    }
}


AGENT_CONFIG = namedtuple('AGENT_CONFIG', ['n_actions', 'state_dim'])


class QController(algorithm_template.Base):
    # pylint: disable=invalid-name
    # pylint: disable=too-few-public-methods
    """
    Implements Qlearning algorithm for Cellular network
    """

    def __init__(self, algorithm_config, env, agent_name):
        """
        Declares local variables
        """
        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        self.algorithm_config = algorithm_config

        # Get shape of environment's state
        self.state_dim = self.env.state_dim

        # Populate agent's config
        self.agent_config = AGENT_CONFIG(
            n_actions=self.env.n_actions,
            state_dim=self.state_dim)

        # Load correct agent
        try:
            self.agent = self._load_agent(self.env.env_name, agent_name)
        except exceptions.AgentNotSupported as error:
            self.logger.exception(error)
            raise
        else:
            self.logger.info(
                "Agent: {} is successfully instantiated!".format(agent_name)
            )

        # Load correct Qlearning instance
        try:
            self.q_instance = self._load_q(self.env.env_name)
        except exceptions.AlgorithmNotImplemented as error:
            self.logger.exception(error)
            raise
        else:
            self.logger.info(
                "Qlearning instance: {} is successfully instantiated!".format(
                    self.q_instance.__class__.__name__
                )
            )

    def _load_agent(self, env_name, agent):
        """
        Helper method to load correct agent
        """
        if env_name not in QLEARNING_AGENTS:
            raise exceptions.AgentNotSupported(
                "Agent: {} is not supported for {} environment".format(
                    agent, env_name)
            )
        return QLEARNING_AGENTS[env_name][agent](
            self.algorithm_config, self.agent_config)

    def _load_q(self, env_name):
        """
        Helper method to load correct Qlearning instance
        """
        if env_name not in SUPPORTED_ENV:
            raise exceptions.AlgorithmNotImplemented(
                "Qlearning is not supported for {} environment".format(
                    env_name
                )
            )
        return SUPPORTED_ENV[env_name](
            self.algorithm_config, self.env, self.agent
        )

    def _execute(self):
        """
        Main method to execute respective Qlearning instance
        """
        return self.q_instance.execute()
