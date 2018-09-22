#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implements naive Q learning with tabular Q function
"""

import logging
import numpy as np
from collections import defaultdict
from collections import namedtuple
from loadbalanceRL.utils import exceptions
from loadbalanceRL.lib.algorithm.Qlearning.agents import agent_template


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, February 21st 2018, 12:42:05 pm'


CELLULAR_AGENT_ACTION = namedtuple(
    'CELLULAR_AGENT_ACTION', ('action', 'ap_id'))


class QNaiveAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):

        # Make sure actions are provided by the environment
        assert self.n_actions

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # log params
        self.logger.info("Configuration used for the Agent:")
        self.logger.info("episodes: {}".format(self.episodes))
        self.logger.info("alpha: {}".format(self.alpha))
        self.logger.info("gamma: {}".format(self.gamma))
        self.logger.info("epsilon: {}".format(self.epsilon))
        self.logger.info("epsilon_decay: {}".format(self.epsilon_decay))
        self.logger.info("epsilon_min: {}".format(self.epsilon_min))

        # Build tabular Q(s, a) model
        self.model = self._build_model()

    def _build_model(self):
        """
        Implements Q(s, a)
        """
        # Initialize Q(s, a) arbitrarily. Here every state is initialized
        # to 0
        return defaultdict(lambda: np.zeros(self.n_actions))

    def _take_action(self, state):
        """
        Implements how to take actions when provided with a state

        This follows epsilon-greedy policy (behavior policy)

        Args
        ----
            state: (tuple)

        Returns
        -------
            action: (float)
        """
        # explore if random number between [0, 1] is less than epsilon,
        # that is this agent exlores 10% of the time and rest exploits
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(range(self.n_actions)))
        return np.argmax(self.model[state])

    def _learn(self, state, action, reward, next_state):
        """
        Implements how the agent learns

        Args
        ----
            state: (tuple)
                Current state of the environment.
            action: (float)
                Current action taken by the agent.
            reward: (float):
                Reward produced by the environment.
            next_state: (tuple)
                Next state of the environment.

        """
        # update epsilon to reduce exploration with increase in episodes
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        target = reward + self.gamma * max(self.model[next_state])
        error = target - self.model[state][action]

        # update
        self.model[state][action] += self.alpha * error

    @property
    def Q(self):
        """
        Public method that keeps Q(s, a) values
        """
        if not self.model:
            raise exceptions.AgentMethodNotImplemented(
                "_model is not implemented for this agent!")
        return self.model


class QCellularAgent(agent_template.Base):
    def __init__(self, alg_config, agent_config):

        # Make sure actions are provided by the environment
        assert self.n_actions

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # log params
        self.logger.info("Configuration used for the QCellular Agent:")
        self.logger.info("episodes: {}".format(self.episodes))
        self.logger.info("alpha: {}".format(self.alpha))
        self.logger.info("gamma: {}".format(self.gamma))
        self.logger.info("epsilon: {}".format(self.epsilon))
        self.logger.info("epsilon_decay: {}".format(self.epsilon_decay))
        self.logger.info("epsilon_min: {}".format(self.epsilon_min))

        # Build tabular Q(s, a) model
        self.model = self._build_model()
        self.ap_model = self._build_ap_model()

    def _build_model(self):
        """
        Implements Q(s, a)
        """
        # Initialize Q(s, a) arbitrarily. Here every state is initialized
        # to 0
        return defaultdict(lambda: np.zeros(self.n_actions))

    def _build_ap_model(self):
        """
        Implements Q(s, stay) for APs only
        """
        return defaultdict(float)

    def get_max_action(self, network_state, ap_list):
        """
        Helper to return action with max Q value

        If there are no neighboring_aps, then action defaults to "stay"
        else return the max with max Q.

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
        # Default action=Stay and ap_id is current_ap
        self.logger.debug("Finding action with max Qvalue!")
        max_action = 0
        ap_id = ap_list[0]

        # Check if the UE has neighboring APs.
        if len(ap_list) > 1:
            self.logger.debug(
                "Q[network_state]: {}".format(self.model[network_state]))

            max_action = np.argmax(self.model[network_state])
            if max_action == 1:
                # Change action to -1, to indicate next_best_ap must
                # be calculated.
                max_action = -1
        max_action_info = CELLULAR_AGENT_ACTION(action=max_action, ap_id=ap_id)
        self.logger.debug(
            "max_action_info from argmax on Q[network_state]: {}".format(
                max_action_info))
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

    def _take_action(self, network_state, ap_list, prob, seed=None):
        """
        Implements how to take actions when provided with a state

        This follows epsilon-greedy policy (behavior policy)

        Args
        ----
            network_state: (tuple)
                State of the network.
            ap_list: (list)
                list of all neighboring aps for the UE.

        Returns
        -------
            action: (int)
                0 (stay) or 1(handoff)
            ap_id: (int)
                id of the next ap
        """
        # explore if random number between [0, 1] is less than epsilon,
        # that is this agent exlores 10% of the time and rest exploits

        if prob < self.epsilon:
            return self.get_random_action(ap_list, seed)
        return self.get_max_action(network_state, ap_list)

    def max_q_for_state(self, state):
        """
        Helper method to fetch the max Q value of the state

        Args
        ----
            state: (tuple)
                State of the environment.


        Returns
        -------
            value: (float)
                Value of action with max Q-value for the given state.
        """
        return np.max(self.model[state])

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

    def _learn(self, state, action, reward, next_state, ue_ap_state):
        """
        Implements how the agent learns

        Args
        ----
            state: (tuple)
                Current network_state of the environment.
            action: (float)
                Current action taken by the agent.
            reward: (float)
                Reward produced by the environment.
            next_state: (tuple)
                Next network_state of the environment.
            ue_ap_state: (tuple)
                State for second Qtable.

        """
        # update epsilon to reduce exploration with increase in episodes
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update Q for the entire network
        target = reward + self.gamma * self.max_q_for_state(next_state)
        error = target - self.model[state][action]

        self.logger.debug(
            "Q table before: {}".format(self.model[state]))
        self.logger.debug("Updating new Q value for the Entire network!")
        self.model[state][action] += self.alpha * error
        self.logger.debug(
            "Q table after: {}".format(self.model[state]))

        if ue_ap_state:
            # update Second Q table
            second_q_target = reward
            second_q_error = second_q_target - self.ap_model[ue_ap_state]

            self.logger.debug(
                "second Qtable before: {}".format(
                    self.ap_model[ue_ap_state]))
            self.logger.debug("Updating new Q value for the second Qtable!")
            self.ap_model[ue_ap_state] += self.alpha * second_q_error
            self.logger.debug(
                "second Qtable after: {}".format(
                    self.ap_model[ue_ap_state]))

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
