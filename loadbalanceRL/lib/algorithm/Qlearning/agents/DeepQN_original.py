#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module stores implememtation of deep Q-Network with experience replay.
"""

import logging
import numpy as np
from collections import defaultdict, namedtuple
import tensorflow as tf
from loadbalanceRL.lib.algorithm.Qlearning.agents import agent_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__author__ = 'Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Monday, July 23rd 2018, 3:53:21 pm'

CELLULAR_AGENT_ACTION = namedtuple(
    'CELLULAR_AGENT_ACTION', ('action', 'ap_id'))


class DQNCellularAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):

        # Make sure actions are provided by the environment
        assert self.n_actions

        # Make sure state_dim is provided by the environment
        assert self.state_dim

        # get hyperparams
        self.learning_rate = 0.05
        self.l1_hidden_units = self.alg_config['L1_HIDDEN_UNITS']
        self.l2_hidden_units = self.alg_config['L2_HIDDEN_UNITS']
        self.l1_activation = self.alg_config['L1_ACTIVATION']
        self.l2_activation = self.alg_config['L2_ACTIVATION']
        # self.loss = self.alg_config['LOSS_FUNCTION']

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "Neural network instance for cellular network is created!")

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
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.state_dim * 2 + 2))

        # builf the target net and the evaluate net
        self.model = self._build_model()
        target_params = tf.get_collection('target_net_params')
        eval_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]

        # For the second Q table
        self.ap_model = self._build_ap_model()
        self.cost_each_step = []
       

    def _build_model(self):
        """
        Helper to build the two networks model, one is the evaluate_net,
        the other is the target net, which have the same structure.
        """
        # build evaluate_net
        self.State = tf.placeholder(tf.float32, [None, self.state_dim], name='state')
        self.Target = tf.placeholder(tf.float32, [None, self.n_actions], name="Q_target")
        n_l1 = self.l1_hidden_units
        n_l2 = self.l2_hidden_units
    
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names_eval= ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            out = self.State
            out = tf.layers.dense(out, units=n_l1, activation=tf.nn.relu)
            out = tf.layers.dense(out, units=n_l2, activation=tf.nn.relu)
            self.y_ = tf.layers.dense(out, self.n_actions)
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.y_ - self.Target))

        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # scope_variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            clipped_grads_and_vars = [(tf.clip_by_norm(item[0],1),item[1]) for item in grads_and_vars]
            self._train_op = optimizer.apply_gradients(clipped_grads_and_vars)    
            # self._train_op = tf.train.AdamOptimizer(
            #     self.learning_rate).minimize(self.loss
            # )
        # build target_net which has the same structure as the evaluation net
        self.Next_State = tf.placeholder(tf.float32, [None, self.state_dim], name='next_state')

        with tf.variable_scope('target_net'):
            # placeholder for next state
            c_names_target = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            out = tf.layers.dense(self.Next_State, units=n_l1, activation=tf.nn.relu)
            out = tf.layers.dense(out, units=n_l2, activation=tf.nn.relu)
            self.q_next= tf.layers.dense(out, units=self.n_actions)

        # start tensorflow session
        self.sess = tf.Session()
        # $ tensorboard --logdir=logs
        tf.summary.FileWriter("logs/", self.sess.graph)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        
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
            action_values = self.predict(np.reshape(network_state, (1, self.state_dim)))
            max_action = np.argmax(action_values)
            if max_action == 1:
                # Change action to -1, to indicate next_best_ap must
                # be calculated.
                max_action = -1
        max_action_info = CELLULAR_AGENT_ACTION(action=max_action, ap_id=ap_id)
        return max_action_info

    def predict(self, state):
        """
        Helper method to predict in the evaluation net
        """
        return self.sess.run(
            self.y_, feed_dict={self.State: state}
        )

    def store_transition(self, state, action, reward, next_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        state = np.reshape(state, (1, self.state_dim))
        next_state = np.reshape(next_state, (1, self.state_dim))
        # the form of the memory
        transition = np.hstack((state[0], [action, reward], next_state[0]))
        # transition={
        #     'state':state, 'action':action,
        #     'reward':reward,'next_state':next_state}
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def _learn(self, state, action, reward, next_state, ue_ap_state):
   
        self.store_transition(state, action, reward, next_state)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_net_params_replaced\n')

        # sample batch memory from all the memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        # q_next comes from the target net, y_ from the evaluate net
        q_next, y_ = self.sess.run(
            [self.q_next, self.y_],
            feed_dict={
                self.Next_State: batch_memory[:, -self.state_dim:],
                self.State: batch_memory[:, :self.state_dim]}
            )
        # change q_target w.r.t q_eval's action
        target = y_.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # action took in the eval_net
        action_index = batch_memory[:, self.state_dim].astype(int)
        reward = batch_memory[:, self.state_dim+1]

        target[batch_index, action_index] = reward + self.gamma * np.max(q_next, axis=1)

        self.learn_step_counter += 1
        # train the eval net
        self.train(batch_memory[:, :self.state_dim], target)

    def train(self, state, target):
        """
        Helper method to train the model, here train the eval net
        """
        # for i in range(100):
        #     _, self.cost = self.sess.run(
        #         [self._train_op, self.loss],
        #         feed_dict={self.State: state,
        #         self.Target: target}
        #         )
        self.cost = 1
        for i in range(2):
            _,self.cost= self.sess.run(
                [self._train_op,self.loss],
                feed_dict={self.State: state,
                self.Target: target}
            )
            
        # print(self.cost)
        self.cost_each_step.append(self.cost)
        self.Q = self.cost_each_step

    @property
    def Q_ap(self):
        """
        Public method that keeps Q_ap(s) values
        """
        return self.ap_model
