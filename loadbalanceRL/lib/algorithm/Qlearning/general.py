#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementation of Qlearning for General environments.
"""

import logging
import numpy as np
import progressbar
from collections import namedtuple


__author__ = 'Ari Saha and Steven Gemelos'
__date__ = 'Sunday, February 18th 2018, 12:21:14 pm'


RESULTS = namedtuple('RESULTS', ['Q', 'Rewards'])


class QlearningForGeneral:
    # pylint: disable=invalid-name
    # pylint: disable=too-few-public-methods
    """
    Implements Qlearning algorithm
    """

    def __init__(self, algorithm_config, env, agent):
        """
        Declares local variables
        """
        self.algorithm_config = algorithm_config
        self.episodes = self.algorithm_config['EPISODES']
        self.env = env
        self.agent = agent

        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self):
        """
        Main method to execute Qlearning algorithm
        """
        self.logger.info("Running qlearning!")

        # Keep track of rewards per episodes
        self.episode_stats = np.zeros(self.episodes)

        # Track progress
        progress_bar = progressbar.ProgressBar(
            maxval=self.episodes,
            widgets=[progressbar.Bar('=', '[', ']'), ' ',
                     progressbar.Percentage()])
        progress_bar.start()

        # Keep generating experience
        for episode in range(self.episodes):
            # get a starting state from the env
            state = self.env.reset()

            while True:

                # Take a step.
                # Get next action based on current state and e-greedy policy.
                action = self.agent.take_action(state)

                # Find next state as a result of current action
                next_state, reward, stop = self.env.step(state, action)
                self.episode_stats[episode] += reward

                # update agent's Q values
                self.agent.learn(state, action, reward, next_state)

                if stop:
                    break
                state = next_state

                # update progress_bar
                progress_bar.update(episode)

        # Exit progress_bar
        progress_bar.finish()

        return RESULTS(Q=self.agent.Q, Rewards=self.episode_stats)
