#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test case for Qlearning general """

import pytest
from collections import OrderedDict
from tests.sample_files import sample_env
from loadbalanceRL import constants
from loadbalanceRL.lib.algorithm.Qlearning import agents
from loadbalanceRL.lib.algorithm.Qlearning import controller
from loadbalanceRL.lib.algorithm.Qlearning import general

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


AGENT_NAME = 'Naive'

QLEARNING_CONFIG = OrderedDict(
    EPISODES=100,
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
    return sample_env.SampleGeneralEnv(constants.SAMPLE_ENV_CONFIG)


@pytest.fixture
def QController_general(sample_env_instance):
    """
    Create an instance of Qcontroller
    """
    return controller.QController(
        QLEARNING_CONFIG, sample_env_instance, AGENT_NAME)


def test_QController_general(QController_general):
    """
    Test for checking if Qcontroller if dispatching correct algorithm and
    respective agents.
    """
    assert isinstance(
        QController_general.agent,
        agents.tabular_q_learning.QNaiveAgent)
    # check if private method _execute is implemented
    assert hasattr(QController_general, '_execute')
    # check if QController is an instance of algorithm_template.Base
    assert hasattr(QController_general, 'execute')
    # check algorithm params
    assert QController_general.episodes == 100
    assert QController_general.alpha == 0.1
    assert QController_general.gamma == 0.8
    assert QController_general.epsilon == 0.3
    assert QController_general.epsilon_decay == 0.99
    assert QController_general.epsilon_min == 0.01
    assert not QController_general.verbose

    results = QController_general.execute()
    assert isinstance(results, general.RESULTS)
