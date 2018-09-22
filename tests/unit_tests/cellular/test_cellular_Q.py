#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for applying Qalgorithm on Cellular environment """

import pytest
from collections import OrderedDict
from tests.sample_files import sample_cellular_env
from tests.sample_files import sample_cellular_q_agent as agent
from loadbalanceRL import constants
from loadbalanceRL.lib.algorithm.Qlearning import cellular

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Monday, April 16th 2018, 4:15:25 pm'

AGENT_NAME = 'Naive'

QLEARNING_CONFIG = OrderedDict(
    EPISODES=1,
    ALPHA=0.1,
    GAMMA=0.8,
    EPSILON=0.3,
    EPSILON_DECAY=0.99,
    EPSILON_MIN=0.01,
    VERBOSE=False,
)


@pytest.fixture
def sample_env_instance():
    """
    Sample env for testing
    """
    return sample_cellular_env.SampleCellularEnv(
        constants.CELLULAR_MODEL_CONFIG)


@pytest.fixture
def cellular_q_agent():
    return agent.SampleAgent()


@pytest.fixture
def QCellular(sample_env_instance, cellular_q_agent):
    """
    Create an instance of Qcontroller
    """
    return cellular.QlearningForCellular(
        QLEARNING_CONFIG, sample_env_instance, cellular_q_agent)


def test_QCellular(QCellular):
    """
    Test for checking if Qcontroller if dispatching correct algorithm and
    respective agents.
    """
    # check algorithm params
    assert QCellular.episodes == 1


def test_QCellular_execute(QCellular):
    """
    Test execute() method for QlearningForCellular
    """

    results = QCellular.execute()
    assert isinstance(results, cellular.CELLULAR_RESULTS)
