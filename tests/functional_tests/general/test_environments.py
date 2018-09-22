#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for environment module """

import pytest
import requests
from collections import OrderedDict
from loadbalanceRL.lib.environment import cellular
from loadbalanceRL.lib.environment.cellular import dev


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


CELLULAR_DEV_CONFIG = OrderedDict(
    NAME='Cellular',
    TYPE='Dev',
    SERVER='0.0.0.0',
    SERVER_PORT='8000',
    VERBOSE=True
)


def server():
    URL = "http://{}:{}".format(
        CELLULAR_DEV_CONFIG['SERVER'], CELLULAR_DEV_CONFIG['SERVER_PORT']
    )
    INDEX = URL + "/"
    try:
        requests.get(INDEX)
    except requests.exceptions.ConnectionError:
        return False
    else:
        return True


@pytest.fixture
def cellular_dev_client():
    """
    Instantiate Dev client for cellular env
    """
    return cellular.base.initialize_client(CELLULAR_DEV_CONFIG)


@pytest.fixture
def cellular_dev_instance(cellular_dev_client):
    """
    Create an instance of Cellular_Dev model
    """
    return cellular.base.CellularNetworkEnv(
        CELLULAR_DEV_CONFIG, cellular_dev_client)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_cellular_dev_env(cellular_dev_instance):
    """
    Test for checking if Cellular_Dev environment is initialized with
    correct config
    """
    assert list(cellular_dev_instance.actions.keys()) == [0, 1]
    assert isinstance(
        cellular_dev_instance._client, dev.client.CellularDevClient)

    print("STATE_DIM:", cellular_dev_instance.state_dim)
    print("AP_DICT:", cellular_dev_instance.ap_dict)
    print("REVERSE_AP_DICT:", cellular_dev_instance._reverse_ap_lookup)
    ue_dict = cellular_dev_instance.ue_dict
    print("UE_DICT:", ue_dict)
    rewards_stats = []
    for _, ue in ue_dict.items():
        state = cellular_dev_instance.get_network_state(ue, ue.ap)
        print("STATE:", state)
        action = 1
        # make a step
        print("Taking next step")
        next_state, reward = cellular_dev_instance.step(
            state, action, ue, ue.ap)
        print("NEXT_STATE:", next_state)
        print("REWARDS:", reward)
        rewards_stats.append(reward)
    print("UE_SLA_STATS:", cellular_dev_instance.ue_sla_stats)
    print("reward_stats:", rewards_stats)
