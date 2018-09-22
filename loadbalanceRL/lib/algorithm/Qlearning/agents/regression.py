#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module stores implementation of various regression models.
"""

import logging
import numpy as np
import tensorflow as tf
from loadbalanceRL.lib.algorithm.Qlearning.agents import agent_template


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


class QLinearRegressionAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):

        # Make sure actions are provided by the environment
        assert self.n_actions

        # Make sure state_dim is provided by the environment
        assert self.state_dim

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

        # Build Linear Regression model
        self._build_model()

    def _build_model(self):
        """
        Helper method to build a model for the agent
        """
        self.State = tf.placeholder(tf.float32, [None, self.state_dim])
        self.Target = tf.placeholder(tf.float32, [None, 1])
        self.W = tf.Variable(tf.ones([self.state_dim, 1]))
        self.b = tf.Variable(tf.ones([1]))

        self.y_ = tf.add(tf.matmul(self.State, self.W), self.b)
        self.cost = tf.reduce_mean(tf.square(self.y_ - self.Target))
        self.training_step = tf.train.GradientDescentOptimizer(
            self.alpha).minimize(self.cost)

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

        # Build Linear Regression model
        self._build_model()

    def _build_model(self):
        """
        Helper method to build a model for the agent
        """
        self.State = tf.placeholder(tf.float32, [None, self.state_dim])
        self.Target = tf.placeholder(tf.float32, [None, 1])
        self.W = tf.Variable(tf.ones([self.state_dim, 1]))
        self.b = tf.Variable(tf.ones([1]))

        self.y_ = tf.add(tf.matmul(self.State, self.W), self.b)
        self.cost = tf.reduce_mean(tf.square(self.y_ - self.Target))
        self.training_step = tf.train.GradientDescentOptimizer(
            self.alpha).minimize(self.cost)

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
