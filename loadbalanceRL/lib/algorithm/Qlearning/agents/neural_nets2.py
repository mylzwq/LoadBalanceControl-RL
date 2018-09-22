#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module stores implememtation of various neural network models.
"""

import logging
import numpy as np
from collections import defaultdict, namedtuple
from keras.layers import Dense, Activation

# pylint: disable=unused-import
from keras.optimizers import Adam  # noqa
from keras.models import Sequential
from loadbalanceRL.lib.algorithm.Qlearning.agents import agent_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__author__ = 'Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, July 21st 2018, 3:53:21 pm'

CELLULAR_AGENT_ACTION = namedtuple(
    'CELLULAR_AGENT_ACTION', ('action', 'ap_id'))


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
        self.learning_rate = self.alg_config['LEARNING_RATE']
        self.l1_hidden_units = self.alg_config['L1_HIDDEN_UNITS']
        self.l2_hidden_units = self.alg_config['L2_HIDDEN_UNITS']
        self.l1_activation = self.alg_config['L1_ACTIVATION']
        self.l2_activation = self.alg_config['L2_ACTIVATION']
        self.loss = self.alg_config['LOSS_FUNCTION']
        self.optimizer = eval(self.alg_config['OPTIMIZER'])(self.learning_rate)

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
        self.ap_model = self._build_ap_model()

    def _build_model(self):
        """
        Implements Q(s, a)
        """
        model = Sequential()
        model.add(Dense(self.l1_hidden_units,
                        input_dim=self.state_dim,
                        activation=self.l1_activation,
                        kernel_initializer='random_normal'))

        model.add(Activation('relu'))

        model.add(Dense(self.l2_hidden_units,
                        activation=self.l2_activation))
        model.add(Activation('relu'))

        model.add(Dense(self.n_actions, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=self.optimizer)
        # model.compile(loss=self.loss,
                    # optimizer=self.optimizer)
        return model

    def _build_ap_model(self):
        """
        Implements Q(s, stay) for APs only
        """
        return defaultdict(float)

    def _take_action(self, network_state, ap_list, prob, seed=None):
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
        network_state = np.reshape(network_state, (1, self.state_dim))
        if prob < self.epsilon:
            return self.get_random_action(ap_list, seed)
        return self.get_max_action(network_state, ap_list)

    def q_from_ap_model(self, state):
        """
        Helper method to fetch the Q value of the state during stay

        Args
        ----
            state: (tuple)
                State of the environment.


        Returns
        -------
            value: (float)
                Q Value of stay for the given state.
        """

        return self.ap_model[state]

    def get_max_action(self, network_state, ap_list):
        """
        Helper to return action with max Q value

        If there are no neighboring_aps, then action defaults to "stay"
        else return the max with max Q. The first step it to max the 
        approximate Q value to decide to "stay" or "handoff", 
        the second step is based the second Q table based on the ue_ap_state

        Args
        ----
            network_state: (tuple)
                State of the network.
            ap_list: (list)
                list containing current ap and neighboring aps

        Returns
        -------
            action: (int)
                0 (stay), 1(handoff)
            ap_id: (int)
                id of the next ap
        """
        max_action = 0
        ap_id = ap_list[0]
        # action_values = np.zeros(1,self.n_actions)
        if len(ap_list) > 1:
            # the values on all the actions
            action_values = self.model.predict(np.reshape(network_state, (1, self.state_dim)))
            max_action = np.argmax(action_values)
            if max_action == 1:
                # Change action to -1, to indicate next_best_ap must
                # be calculated.
                max_action = -1
        max_action_info = CELLULAR_AGENT_ACTION(action=max_action, ap_id=ap_id)
        return max_action_info

    def get_random_action(self, ap_list, seed=None):
        """
        Helper to return a random action

        Args
        ----
            ap_list: (list)
                list containing current ap and neighboring aps

        Returns
        -------
            action: (int)
                0 (stay), 1(handoff)
            ap_id: (int)
                id of the next ap
        """
        # Default action=Stay and ap_id is current_ap
        self.logger.debug("Taking a random action!")

        # Seed is used for testing purposes only!
        if seed:
            np.random.seed(seed)

        random_action = 0
        ap_id = ap_list[0]

        # Check if the UE has neighboring APs.
        if len(ap_list) > 1:
            random_action = np.random.choice(self.n_actions)
            if random_action == 1:
                ap_id = np.random.choice(ap_list[1:])
        random_action_info = CELLULAR_AGENT_ACTION(
            action=random_action, ap_id=ap_id)
        self.logger.debug(
            "random_action_info: {}".format(random_action_info)
        )
        return random_action_info

    def _learn(self, state, action, reward, next_state, ue_ap_state=None):
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
        next_state = np.reshape(next_state, (1, self.state_dim))
        # newQ is Q(s',a) for the prediction of next_state
        newQ = self.model.predict(next_state, batch_size=1)
        target = self.model.predict(
            state, batch_size=1, verbose=self.verbose)[0]

        target[action] = reward + self.gamma * np.max(newQ)

        target = np.reshape(target, (1, self.n_actions))
        # fit the model with the fixed target 
        self.model.fit(
            state, target, epochs=1, batch_size=1,verbose=1)

        if ue_ap_state:
            # update Second Q table
            second_q_target = reward
            second_q_error = second_q_target - self.ap_model[ue_ap_state]
            self.logger.debug("Updating second Q table in regression method!")
            # update the second table for if the ue take handoff
            self.ap_model[ue_ap_state] += self.alpha * second_q_error

    @property
    def Q(self):
        """
        Public method that keeps Q(s, a) values
        """
        return self.model

    @property
    def Q_ap(self):
        """
        Public method that keeps Q_ap(s) values
        """
        return self.ap_model
