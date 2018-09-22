#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module stores implememtation of deep Q-Network with experience replay.
"""

import logging
import numpy as np
import random
from collections import defaultdict, namedtuple
import tensorflow as tf
from keras.optimizers import Adam  # noqa
from keras.models import Sequential
from keras.layers import Dense, Activation
from collections import deque
from loadbalanceRL.lib.algorithm.Qlearning.agents import agent_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__author__ = 'Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Monday, July 24th 2018, 3:53:21 pm'

CELLULAR_AGENT_ACTION = namedtuple(
    'CELLULAR_AGENT_ACTION', ('action', 'ap_id'))


class DQNCellularAgent(agent_template.Base):
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
        self.optimizer = eval(self.alg_config['OPTIMIZER'])(self.alpha)
        self.loss = self.alg_config['LOSS_FUNCTION']
        self.optimizer = eval(self.alg_config['OPTIMIZER'])(self.learning_rate)

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "Deep Q-Network instance for cellular network is created!")

        # log params
        self.logger.info("Configuration used for the Agent in DQN:")
        self.logger.info("episodes: {}".format(self.episodes))
        self.logger.info("alpha: {}".format(self.alpha))
        self.logger.info("gamma: {}".format(self.gamma))
        self.logger.info("epsilon: {}".format(self.epsilon))
        self.logger.info("epsilon_decay: {}".format(self.epsilon_decay))
        self.logger.info("epsilon_min: {}".format(self.epsilon_min))

        # For the two network configuration:
        self.replace_target_iter = self.alg_config['REPLACE_TARGET_ITER']
        self.memory_size = self.alg_config['MEMORY_SIZE']
        self.batch_size = self.alg_config['BATCH_SIZE']
        self.logger.info("After {} steps to replace target net".format(self.replace_target_iter))
        self.logger.info("Memory Size: {}".format(self.memory_size))
        self.logger.info("Batch Size: {}".format(self.batch_size))

        # used to check when to replace the target parameters
        # as well as to decide when to learn
        self.learn_step_counter = 0
        # initialize zero memory [s, a, r, s']
        # the memory buffer is used for experience replay
        self.memory  = deque(maxlen=self.memory_size)

        # builf the target net and the evaluate net
        # they have the same structure
        self.model = self._build_model()
        self.target_model = self._build_model()

        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

        # For the second Q table
        self.ap_model = self._build_ap_model()

       
    def replace_target_op(self):
        """
        Helper to replace the target_net parameters with the evaluation network
        """
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def _build_model(self):
        """
        Helper to build the two networks model, one is the evaluate_net,
        the other is the target net, which have the same structure.
        """
        # build evaluate_net
        # self.State = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
        # self.Target = tf.placeholder(tf.float32, [None, self.n_actions], name="Q_target")
        # Creat the model
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
        return model

        
    def _build_ap_model(self):
        """
        Implements Q(s, stay) for APs only
        """
        return defaultdict(float)

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
        if prob < self.epsilon:
            return self.get_random_action(ap_list, seed)
        return self.get_max_action(network_state, ap_list)

    def get_max_action(self, network_state, ap_list):
        """
        Helper to return action with max Q value

        If there are no neighboring_aps, then action defaults to "stay"
        else return the max with max Q. The first step it to max the approximate
        Q value to decide to "stay" or "handoff", the second step is based the
        second Q table based on the ue_ap_state

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


    def store_transition(self, state, action, reward, next_state):
        """
        Help to store the information of 
        [tate, action, reward, next_state] into memory buffer
        """
        
        state = np.reshape(state, (1, self.state_dim))
        next_state = np.reshape(next_state, (1, self.state_dim))
        # the form of the memory
        transition = np.hstack((state[0], action, reward, next_state[0]))
        self.memory.append(transition)

    def replay(self):
        """
        Help to execute the experience replay and fixed Q target.
        """
        if len(self.memory) < self.batch_size: 
            return
        # sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        # samples = self.memory[sample_index]
        samples=random.sample(self.memory, self.batch_size)

        for sample in samples:
            state = sample[:self.state_dim]
            next_state= sample[ -self.state_dim:]
            reward = sample[self.state_dim+1]
            action = sample[self.state_dim].astype(int)
            target = self.target_model.predict(np.reshape(state, (1, self.state_dim)))
            Q_future = max(self.target_model.predict(np.reshape(next_state, (1, self.state_dim)))[0])
            target[0][action] = reward + Q_future * self.gamma
            self.model.fit(np.reshape(state, (1, self.state_dim)), target, epochs=1, batch_size=1, verbose=0)

    def _learn(self, state, action, reward, next_state, ue_ap_state):
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
        
        self.store_transition(state, action, reward, next_state)
        # every fixed time steps, update the parameters in target 
        # net with that from the evaluation net
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.logger.debug(
            "Replace the target net's parameters at the {} time step"
            .format(self.learn_step_counter )
            )
            self.replace_target_op()
             
        self.learn_step_counter += 1

        state = np.reshape(state, (1, self.state_dim))
        next_state = np.reshape(next_state, (1, self.state_dim))
        #update the target value in the epsion-greedy action
        newQ = self.target_model.predict(next_state, batch_size=1)
        target = self.target_model.predict(
            state, batch_size=1, verbose=0)[0]

        target[action] = reward + self.gamma * np.max(newQ)
        target = np.reshape(target, (1, self.n_actions))
        # train the eval net
        self.model.fit(
            state, target, epochs=1, batch_size=1,verbose=0)
        self.Q = self.model.get_weights()

    @property
    def Q_ap(self):
        """
        Public method that keeps Q_ap(s) values
        """
        return self.ap_model
