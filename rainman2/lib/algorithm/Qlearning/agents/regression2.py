#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module stores implementation of various regression models.
"""

import logging
from collections import defaultdict
from collections import namedtuple
import numpy as np
import tensorflow as tf
from loadbalanceRL.lib.algorithm.Qlearning.agents import agent_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__author__ = 'Mingyang Liu (liux3941@umn.edu)'
__date__ = 'Tuesday, July 20th 2018, 3:57:18 pm'


CELLULAR_AGENT_ACTION = namedtuple(
    'CELLULAR_AGENT_ACTION', ('action', 'ap_id'))


class QLinearRegressionAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):

        # Make sure actions are provided by the environment
        assert self.n_actions

        # Make sure state_dim is provided by the environment
        assert self.state_dim
        self.learning_rate= self.alg_config['LEARNING_RATE']

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Linear regression instance is created!")

        # log params
        self.logger.info("Configuration used for the Agent:")
        self.logger.info("episodes: {}".format(self.episodes))
        self.logger.info("alpha: {}".format(self.alpha))
        self.logger.info("gamma: {}".format(self.gamma))
        self.logger.info("epsilon: {}".format(self.epsilon))
        self.logger.info("epsilon_decay: {}".format(self.epsilon_decay))
        self.logger.info("epsilon_min: {}".format(self.epsilon_min))
        self.logger.info("learning_rate: {}".format(self.learning_rate))

        # Build Linear Regression model
        self._build_model()

    def _build_model(self):
        """
        Helper method to build a model for the agent
        """
        self.State = tf.placeholder(tf.float32, [None, self.state_dim])
        self.Target = tf.placeholder(tf.float32, [None, 1])
        self.W = tf.Variable(tf.ones([self.state_dim, self.n_actions]))
        self.b = tf.Variable(tf.ones([self.n_actions]))
        # prediction of the Q value
        self.y_ = tf.add(tf.matmul(self.State, self.W), self.b)
        # loss function 
        self.cost = tf.reduce_mean(tf.square(self.Target-self.y_))

        self.training_step = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.cost)


        # start tensorflow session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def predict(self, state):
        """
        Helper method to predict models's output
        """
        return self.sess.run(
            self.y_, feed_dict={self.State: state})

    def train(self, state, target):
        """
        Helper method to train the model
        """
        self.sess.run(
            self.training_step,
            feed_dict={self.State: state, self.Target: target})

    def model_cost(self, state, target):
        """
        Calculate cost
        """
        return self.sess.run(
            self.cost,
            feed_dict={self.State: state, self.Target: target})

    def model_error(self, pred_y, test_y):
        """
        Calculate mean sqare error
        """
        return tf.reduce_mean(tf.square(pred_y - test_y))

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
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(range(self.n_actions)))
        return np.argmax(self.predict(np.reshape(state, (1, self.state_dim))))

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
        target = self.predict(state)
        self.train(state, target)


class QCellularLinearRegressionAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):
        # Make sure actions are provided by the environment
        assert self.n_actions

        # Make sure state_dim is provided by the environment
        assert self.state_dim
        self.learning_rate= self.alg_config['LEARNING_RATE']

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "Linear regression instance for cellular network is created!")

        # log params
        self.logger.info("Configuration used for the Agent:")
        self.logger.info("episodes: {}".format(self.episodes))
        self.logger.info("alpha: {}".format(self.alpha))
        self.logger.info("gamma: {}".format(self.gamma))
        self.logger.info("epsilon: {}".format(self.epsilon))
        self.logger.info("epsilon_decay: {}".format(self.epsilon_decay))
        self.logger.info("epsilon_min: {}".format(self.epsilon_min))
        self.logger.info("learning_rate: {}".format(self.learning_rate))

        # Build Linear Regression model
        
        self._build_model()
        self.ap_model = self._build_ap_model()
        self.l = 0
        self.train_step = 0

    def _build_model(self):
        """
        Helper method to build a model for the agent
        """
        # self.learning_rate = 0.01
        self.State = tf.placeholder(tf.float32, [None, self.state_dim])
        self.Target = tf.placeholder(tf.float32, [None, self.n_actions])
        self.W = tf.Variable(tf.random_normal([self.state_dim, self.n_actions], stddev=0.1))
        # self.W = tf.Variable(tf.ones([self.state_dim, self.n_actions]))
        self.b = tf.Variable(tf.random_normal([self.n_actions], stddev=0.1))

        self.y_ = tf.add(tf.matmul(self.State, self.W), self.b)
        self.a = tf.placeholder(tf.int32)
        self.one_hot_mask=tf.one_hot(self.a, self.n_actions)
        self.cost = tf.reduce_mean(tf.square(self.y_ - self.Target)) + \
                    0.1* tf.nn.l2_loss(
                        tf.reduce_sum(tf.multiply(self.W,self.one_hot_mask),axis=1))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # normalize the model
        self.grads_and_vars = self.optimizer.compute_gradients(self.cost)
        self.clipped_grads_and_vars = [(tf.clip_by_norm(item[0],1),item[1]) for item in self.grads_and_vars]
        self.training_step = self.optimizer.apply_gradients(self.clipped_grads_and_vars) 
        
        # self.grads_and_vars = self.optimizer.compute_gradients(self.cost)
        # self.grad_norms = tf.add_n([tf.nn.l2_loss(g) for g, v in self.grads_and_vars])
        
        # self.training_step = tf.train.GradientDescentOptimizer(
            # self.learning_rate).minimize(self.cost)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        tf.summary.FileWriter("logs/linear", self.sess.graph)

    def _build_ap_model(self):
        """
        Implements Q(s, stay) for APs only
        """
        return defaultdict(float)

    def predict(self, state):
        """
        Helper method to predict models's output
        """
        return self.sess.run(
            self.y_, feed_dict={self.State: state})

    def model_cost(self, state, target):
        """
        Calculate cost
        """
        return self.sess.run(
            self.cost,
            feed_dict={self.State: state, self.Target: target})

    def model_error(self, pred_y, test_y):
        """
        Calculate mean sqare error
        """
        return tf.reduce_mean(tf.square(pred_y - test_y))

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
            action_values = self.predict(np.reshape(network_state, (1, self.state_dim)))
            max_action = np.argmax(action_values)
            if max_action == 1:
                # Change action to -1, to indicate next_best_ap must
                # be calculated.
                max_action = -1
        max_action_info = CELLULAR_AGENT_ACTION(action=max_action, ap_id=ap_id)
        return max_action_info

    def _learn(self, state, action, reward, next_state, ue_ap_state):
        """
        Implements how the agent learns and the training step

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
        
        target = self.predict(state)
        target[0][action] = reward + self.gamma * self.max_q_for_state(next_state)

        if ue_ap_state:
            # update Second Q table
            second_q_target = reward
            second_q_error = second_q_target - self.ap_model[ue_ap_state]
            self.logger.debug("Updating second Q table in regression method!")
            # update the second table for if the ue take handoff
            self.ap_model[ue_ap_state] += self.alpha * second_q_error

        # target = self.predict(state)
        self.train_step +=1
        self.train(state, target, action, self.train_step)

    def max_q_for_state(self, network_state):
        """
        Helper method to fetch the max Q value of the network_state
        """
        return np.max(
            self.predict(np.reshape(network_state, (1, self.state_dim)))
            )

    def train(self, state, target, action, train_step):
        """
        Helper method to train the model, including:
        update the loss, update the optimizer, and the gradients
        """
        for i in range(10):
            _, loss= self.sess.run(
                [self.training_step, self.cost],
                feed_dict={self.State: state,
                self.Target: target, self.a: action})
        # self.saver.save(self.sess, 'checkpoints/linear', train_step)

        self.l  += loss
        self.Q = self.sess.run(self.W)
        # print(self.sess.run([self.W,self.b]))

    @property
    def Q_ap(self):
        """
        Public method that keeps Q_ap(s) values
        """
        return self.ap_model
