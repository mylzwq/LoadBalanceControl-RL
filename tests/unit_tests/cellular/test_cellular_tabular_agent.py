#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for cellular_tabular_agent """

import pytest
from collections import OrderedDict, namedtuple
from tests.sample_files import sample_cellular_env as env
from loadbalanceRL.lib.algorithm.Qlearning.agents import tabular_q_learning

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Friday, April 13th 2018, 2:04:45 pm'

QLEARNING_BASIC_CONFIG = OrderedDict(
    EPISODES=100,
    ALPHA=0.2,
    GAMMA=0.9,
    EPSILON=0.3,
    EPSILON_DECAY=0.99,
    EPSILON_MIN=0.01,
    VERBOSE=True,
)

AGENT_CONFIG = namedtuple('AGENT_CONFIG', ['n_actions', 'state_dim'])


@pytest.fixture
def agent():
    """
    Creates QCellularAgent
    """
    agent_config = AGENT_CONFIG(
        n_actions=2,
        state_dim=7
    )
    return tabular_q_learning.QCellularAgent(
        QLEARNING_BASIC_CONFIG, agent_config)


def test_take_action(agent):

    # Test take random action
    assert agent.take_action(
        env.NETWORK_STATE_1_8, [8, 4], 0.2, 5) ==\
        tabular_q_learning.CELLULAR_AGENT_ACTION(action=1, ap_id=4)


def test_learn(agent):
    agent.learn(env.NETWORK_STATE_1_8, 0, -2.5, env.NETWORK_STATE_1_4, None)
    assert agent.Q[env.NETWORK_STATE_1_8].tolist() == [-0.5, 0.]

    agent.learn(env.NETWORK_STATE_2_10,
                1,
                -1.25,
                env.NETWORK_STATE_2_11,
                env.UE_AP_STATE_2_11)
    assert agent.Q[env.NETWORK_STATE_2_10].tolist() == [0., -0.25]

    # Test take max action, which will pick "handoff"
    assert agent.take_action(
        env.NETWORK_STATE_1_8, [8, 4], 0.6) ==\
        tabular_q_learning.CELLULAR_AGENT_ACTION(action=-1, ap_id=8)


def test_max_q_for_state(agent):
    assert agent.max_q_for_state(env.NETWORK_STATE_1_8) == 0.0
