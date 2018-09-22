#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module stores implememtation of various neural network models.
"""

import logging
import numpy as np
from keras.layers import Dense

# pylint: disable=unused-import
from keras.optimizers import Adam  # noqa
from keras.models import Sequential
from loadbalanceRL.lib.algorithm.Qlearning.agents import agent_template


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


class QNNAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):

        # Make sure actions are provided by the environment
        assert self.n_actions

        # Make sure state_dim is provided by the environment
        assert self.state_dim

        # get hyperparams
        self.l1_hidden_units = self.alg_config['L1_HIDDEN_UNITS']
        self.l2_hidden_units = self.alg_config['L2_HIDDEN_UNITS']
        self.l1_activation = self.alg_config['L1_ACTIVATION']
        self.l2_activation = self.alg_config['L2_ACTIVATION']
        self.loss = self.alg_config['LOSS_FUNCTION']
        self.optimizer = eval(self.alg_config['OPTIMIZER'])(self.alpha)

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Basic Neural network instance is created!")

        # log params
        self.logger.info("Configuration used for the Agent:")
        self.logger.info("episodes: {}".format(self.episodes))
        self.logger.info("alpha: {}".format(self.alpha))
        self.logger.info("gamma: {}".format(self.gamma))
        self.logger.info("epsilon: {}".format(self.epsilon))
        self.logger.info("epsilon_decay: {}".format(self.epsilon_decay))
        self.logger.info("epsilon_min: {}".format(self.epsilon_min))

        # Build NN model to estimate Q(s, a)
        self.model = self._build_model()

    def _build_model(self):
        """
        Implements Q(s, a)
        """
        model = Sequential()
        model.add(Dense(self.l1_hidden_units,
                        input_dim=self.state_dim,
                        activation=self.l1_activation,
                        kernel_initializer='random_normal'))
        model.add(Dense(self.l2_hidden_units,
                        activation=self.l2_activation))
        model.add(Dense(self.n_actions, activation='linear'))
        model.summary()
        model.compile(loss=self.loss,
                      optimizer=self.optimizer)
        return model

    def _take_action(self, state):
        """
        Implements how to take actions when provided with a state

        This follows epsilon-greedy policy (behavior policy)

        Args:
            state: (tuple)

        Returns:
            action: (float)
        """
        # explore if random number between [0, 1] is less than epsilon,
        # that is this agent exlores 10% of the time and rest exploits
        state = np.reshape(state, (1, self.state_dim))
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(range(self.n_actions)))
        return np.argmax(self.model.predict(
            state, batch_size=1, verbose=self.verbose)[0])

    def _learn(self, state, action, reward, next_state):
        """
        Implements how the agent learns

        Args:
            state: (tuple)
                Current state of the environment.
            action: (float)
                Current action taken by the agent.
            reward: (float):
                Reward produced by the environment.
            next_state: (tuple)
                Next state of the environment.

        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.reshape(state, (1, self.state_dim))
        target = self.model.predict(
            state, batch_size=1, verbose=self.verbose)[0]

        target[action] = reward + self.gamma * max(target)
        target = np.reshape(target, (1, self.n_actions))
        self.model.fit(
            state, target, epochs=1, batch_size=1, verbose=self.verbose)


class QCellularNNAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):

        # Make sure actions are provided by the environment
        assert self.n_actions

        # Make sure state_dim is provided by the environment
        assert self.state_dim

        # get hyperparams
        self.l1_hidden_units = self.alg_config['L1_HIDDEN_UNITS']
        self.l2_hidden_units = self.alg_config['L2_HIDDEN_UNITS']
        self.l1_activation = self.alg_config['L1_ACTIVATION']
        self.l2_activation = self.alg_config['L2_ACTIVATION']
        self.loss = self.alg_config['LOSS_FUNCTION']
        self.optimizer = eval(self.alg_config['OPTIMIZER'])(self.alpha)

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "Neural network instance for cellular network is created!")

        # log params
        self.logger.info("Configuration used for the Agent:")
        self.logger.info("episodes: {}".format(self.episodes))
        self.logger.info("alpha: {}".format(self.alpha))
        self.logger.info("gamma: {}".format(self.gamma))
        self.logger.info("epsilon: {}".format(self.epsilon))
        self.logger.info("epsilon_decay: {}".format(self.epsilon_decay))
        self.logger.info("epsilon_min: {}".format(self.epsilon_min))

        # Build NN model to estimate Q(s, a)
        self.model = self._build_model()

    def _build_model(self):
        """
        Implements Q(s, a)
        """
        model = Sequential()
        model.add(Dense(self.l1_hidden_units,
                        input_dim=self.state_dim,
                        activation=self.l1_activation,
                        kernel_initializer='random_normal'))
        model.add(Dense(self.l2_hidden_units,
                        activation=self.l2_activation))
        model.add(Dense(self.n_actions, activation='linear'))
        model.summary()
        model.compile(loss=self.loss,
                      optimizer=self.optimizer)
        return model

    def _take_action(self, state):
        """
        Implements how to take actions when provided with a state

        This follows epsilon-greedy policy (behavior policy)

        Args:
            state: (tuple)

        Returns:
            action: (float)
        """
        # explore if random number between [0, 1] is less than epsilon,
        # that is this agent exlores 10% of the time and rest exploits
        state = np.reshape(state, (1, self.state_dim))
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(range(self.n_actions)))
        return np.argmax(self.model.predict(
            state, batch_size=1, verbose=self.verbose)[0])

    def _learn(self, state, action, reward, next_state):
        """
        Implements how the agent learns

        Args:
            state: (tuple)
                Current state of the environment.
            action: (float)
                Current action taken by the agent.
            reward: (float):
                Reward produced by the environment.
            next_state: (tuple)
                Next state of the environment.

        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.reshape(state, (1, self.state_dim))
        target = self.model.predict(
            state, batch_size=1, verbose=self.verbose)[0]

        target[action] = reward + self.gamma * max(target)
        target = np.reshape(target, (1, self.n_actions))
        self.model.fit(
            state, target, epochs=1, batch_size=1, verbose=self.verbose)
