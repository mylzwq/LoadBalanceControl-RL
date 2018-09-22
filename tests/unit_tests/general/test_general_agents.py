#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for agents module """

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from collections import OrderedDict, namedtuple
from loadbalanceRL.lib.algorithm.Qlearning import agents


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


QLEARNING_BASIC_CONFIG = OrderedDict(
    EPISODES=1000,
    ALPHA=0.1,
    GAMMA=0.8,
    EPSILON=0.3,
    EPSILON_DECAY=0.999,
    EPSILON_MIN=0.01,
    VERBOSE=False,
)


QLEARNING_FA_CONFIG = OrderedDict(
    EPISODES=1000,
    ALPHA=0.1,
    GAMMA=0.8,
    EPSILON=0.3,
    EPSILON_DECAY=0.999,
    EPSILON_MIN=0.01,
    VERBOSE=False,
    L1_HIDDEN_UNITS=3,
    L2_HIDDEN_UNITS=2,
    L1_ACTIVATION='softmax',
    L2_ACTIVATION='tanh',
    LOSS_FUNCTION='categorical_crossentropy',
    OPTIMIZER='Adam'
)

# Environment variables
N_ACTIONS = 4
STATE_DIM = 5
np.random.seed(3)
SAMPLE_STATE = np.around(np.random.rand(1, STATE_DIM), decimals=2)
SAMPLE_ACTION = 1
SAMPLE_REWARD = -1
SAMPLE_NEXT_STATE = np.around(np.random.rand(1, STATE_DIM), decimals=2)

AGENT_CONFIG = namedtuple('AGENT_CONFIG', ['n_actions', 'state_dim'])


@pytest.fixture
def naive_agent():
    """
    Create an instance of tabular based agent model
    """
    agent_config = AGENT_CONFIG(
        n_actions=N_ACTIONS,
        state_dim=None,
    )

    return agents.tabular_q_learning.QNaiveAgent(
        QLEARNING_BASIC_CONFIG,
        agent_config)


@pytest.fixture
def regression_agent():
    """
    Create an instance of linear_regression agent model
    """
    agent_config = AGENT_CONFIG(
        n_actions=N_ACTIONS,
        state_dim=STATE_DIM
    )

    return agents.regression.QLinearRegressionAgent(
        QLEARNING_FA_CONFIG,
        agent_config)


@pytest.fixture
def nn_agent():
    """
    Create an instance of NN agent model
    """
    agent_config = AGENT_CONFIG(
        n_actions=N_ACTIONS,
        state_dim=STATE_DIM
    )

    return agents.neural_nets.QNNAgent(
        QLEARNING_FA_CONFIG,
        agent_config)


def test_naive_agent(naive_agent):
    """
    Test for checking if naiveQAgent model is initialized
    with correct config.
    """
    # check if necessary private methods are implemented
    assert hasattr(naive_agent, '_take_action')
    assert hasattr(naive_agent, '_learn')

    # check if QNaiveAgent is an instance of agent_template.Base
    assert hasattr(naive_agent, 'take_action')
    assert hasattr(naive_agent, 'learn')

    # check algorithm params
    assert naive_agent.episodes == 1000
    assert naive_agent.alpha == 0.1
    assert naive_agent.gamma == 0.8
    assert naive_agent.epsilon == 0.3
    assert naive_agent.epsilon_decay == 0.999
    assert naive_agent.epsilon_min == 0.01
    assert not naive_agent.verbose
    assert naive_agent.n_actions == 4


def test_linear_regression_agent(regression_agent):
    """
    Test for checking if linear_regression_agent model is initialized
    with correct config.
    """
    # check if necessary private methods are implemented
    assert hasattr(regression_agent, '_take_action')
    assert hasattr(regression_agent, '_learn')

    # check if qlearning_regression is an instance of agent_template.Base
    assert hasattr(regression_agent, 'take_action')
    assert hasattr(regression_agent, 'learn')

    # check algorithm params
    assert regression_agent.episodes == 1000
    assert regression_agent.alpha == 0.1
    assert regression_agent.gamma == 0.8
    assert regression_agent.epsilon == 0.3
    assert regression_agent.epsilon_decay == 0.999
    assert regression_agent.epsilon_min == 0.01
    assert not regression_agent.verbose
    assert regression_agent.n_actions == 4
    assert regression_agent.state_dim == 5


def test_nn_agent(nn_agent):
    """
    Test for checking if QNNAgent model is initialized
    with correct config.
    """
    # check if necessary private methods are implemented
    assert hasattr(nn_agent, '_take_action')
    assert hasattr(nn_agent, '_learn')

    # check if QNNAgent is an instance of agent_template.Base
    assert hasattr(nn_agent, 'take_action')
    assert hasattr(nn_agent, 'learn')

    # check algorithm params
    assert nn_agent.episodes == 1000
    assert nn_agent.alpha == 0.1
    assert nn_agent.gamma == 0.8
    assert nn_agent.epsilon == 0.3
    assert nn_agent.epsilon_decay == 0.999
    assert nn_agent.epsilon_min == 0.01
    assert not nn_agent.verbose
    assert nn_agent.n_actions == 4
    assert nn_agent.state_dim == 5

    print("Running with sample data!")
    print("STATE:", SAMPLE_STATE[0])
    assert_array_equal(
        SAMPLE_STATE, np.array([[0.55, 0.71, 0.29, 0.51, 0.89]]))

    print("ACTION:", SAMPLE_ACTION)
    assert SAMPLE_ACTION == 1

    print("REWARD:", SAMPLE_REWARD)
    assert SAMPLE_REWARD == -1

    print("NEXT_STATE:", SAMPLE_NEXT_STATE)
    assert_array_equal(
        SAMPLE_NEXT_STATE, np.array([[0.9, 0.13, 0.21, 0.05, 0.44]])
    )

    predicted_action = nn_agent.take_action(SAMPLE_STATE)
    print("predicting next action: {}".format(predicted_action))
    assert predicted_action in range(4)
    print("")

    print("learning!")
    nn_agent.learn(
        SAMPLE_STATE, SAMPLE_ACTION, SAMPLE_REWARD, SAMPLE_NEXT_STATE)

    predicted_action = nn_agent.take_action(SAMPLE_STATE)
    print("predicting next action: {}".format(predicted_action))
    assert predicted_action in range(4)
