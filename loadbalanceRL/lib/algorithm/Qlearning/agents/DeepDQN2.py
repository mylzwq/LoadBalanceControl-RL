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
        self.evalNet = self._build_network("eval_net")
        self.targetNet = self._build_network("target_net")
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)
        
    
    def _build_network(self,scope_name) -> None:
        """DQN Network architecture (simple MLP)
        """
        with tf.variable_scope(scope_name):
            self.State = tf.placeholder(tf.float32, [None, self.state_dim], name="input_state")
            net = self.State

            net = tf.layers.dense(net, self.l1_hidden_units, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.l2_hidden_units,activation=tf.nn.relu)
            net = tf.layers.dense(net, self.n_actions)
            self.Qpre = net

            self.Target = tf.placeholder(tf.float32, shape=[None, self.n_actions])
            self.cost = tf.losses.mean_squared_error(self.Target, self.Qpre)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.cost)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    def predict(self,state):

        return self.sess.run(
            self.Qpre, feed_dict={self.State: state})
    
    def update(self, x_stack, y_stack) :
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        feed = {
            self.State: x_stack,
            self.Target: y_stack
        }
        return self.session.run([self._loss, self._train], feed)
    
    def replay_train(evalDQN, targetDQN, train_batch):
        """Trains `evalDQN` with target Q values given by `targetDQN`
        Args:
            mainDQN : Evaluate DQN that will be trained
            targetDQN : Target DQN that will predict Q_target
            train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s')
            [(state, action, reward, next_state), ...]
        Returns:
            float: After updating `evalDQN`, it returns a `loss`
        """
        states = np.vstack([x[0] for x in train_batch])
        actions = np.array([x[1] for x in train_batch])
        rewards = np.array([x[2] for x in train_batch])
        next_states = np.vstack([x[3] for x in train_batch])
        X = states

        Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1)

        y = mainDQN.predict(states)
        y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
        return mainDQN.update(X, y)

    def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str):
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
        op_holder = []

        src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            op_holder.append(dest_var.assign(src_var.value()))

        return op_holder

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

            if self.learn_step_counter % self.replace_target_iter == 0:
            self.logger.debug(
            "Replace the target net's parameters at the {} time step"
            .format(self.learn_step_counter )
            )
            copy-ops= self.get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
            sess.run(copy_ops)

