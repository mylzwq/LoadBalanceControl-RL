#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for interface """

import pytest
from collections import OrderedDict
from loadbalanceRL import RAINMAN3
from loadbalanceRL.lib import interface
from loadbalanceRL.lib.environment.cellular.dev import client as cellular_dev_client


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


QLEARNING_REGRESSION_CONFIG = OrderedDict(
    EPISODES=1000,
    ALPHA=0.1,
    GAMMA=0.8,
    EPSILON=0.3,
    EPSILON_DECAY=0.999,
    EPSILON_MIN=0.01,
    VERBOSE=False,
)

CELLULAR_DEV_CONFIG = OrderedDict(
    NAME='Cellular',
    TYPE='Dev',
    SERVER='0.0.0.0',
    SERVER_PORT='8000',
    VERBOSE=True
)


@pytest.fixture
def loadbalance_instance():
    """
    Create a Rainman instance
    """
    RAINMAN3.algorithm_config = QLEARNING_REGRESSION_CONFIG
    RAINMAN3.environment_config = CELLULAR_DEV_CONFIG
    return RAINMAN3


def test_build_env_client(loadbalance_instance):
    """
    Tests _build_env_client function
    """
    client = loadbalance_instance._build_env_client('Cellular')
    assert isinstance(client, cellular_dev_client.CellularDevClient)


@pytest.fixture
def test_build_env_instance(loadbalance_instance):
    """
    Tests _build_env_instance function.
    """
    env_instance = loadbalance_instance._build_env_instance('Cellular')
    assert isinstance(env_instance,
                      interface.SUPPORTED_ENVIRONMENTS['Cellular'])
    return env_instance


def test_build_alg_instance(loadbalance_instance, test_build_env_instance):
    """
    Tests _build_alg_instance function.
    """
    alg_instance = loadbalance_instance._build_alg_instance(
        'Qlearning', test_build_env_instance, 'Naive')
    assert isinstance(alg_instance,
                      interface.SUPPORTED_ALGORITHMS['Qlearning'])
    return alg_instance


def main():
    """
    Test locally
    """
    rainman = loadbalance_instance()
    env_instance = test_build_env_instance(rainman)
    test_build_alg_instance(rainman, env_instance)


if __name__ == '__main__':
    main()
