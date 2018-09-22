#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for cellular environment """

import pytest
from collections import OrderedDict
from loadbalanceRL import constants
from loadbalanceRL.lib.environment.cellular import base
from tests.sample_files import sample_cellular_client as client
from tests.sample_files import sample_cellular_env as env

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


# pylint: disable=E1101

CELLULAR_DEV_CONFIF = OrderedDict(
    NAME='Cellular',
    TYPE='Dev',
    SERVER='0.0.0.0',
    SERVER_PORT='8000',
    VERBOSE=True,
)


@pytest.fixture
def dev_client():
    return client.CellularDevClient(constants.CELLULAR_MODEL_CONFIG)


@pytest.fixture
def dev_network(dev_client):
    return base.CellularNetworkEnv(
        constants.CELLULAR_MODEL_CONFIG, dev_client)


def test_ap_dict(dev_network):
    ap_dict = dev_network.ap_dict
    for _id, ap in ap_dict.items():
        assert ap.ap_id == env.AP_DICT[_id].ap_id
        assert ap.location == env.AP_DICT[_id].location
        assert ap.n_ues == env.AP_DICT[_id].n_ues
        assert ap.ues_meeting_sla == env.AP_DICT[_id].ues_meeting_sla


def test_ue_dict(dev_network):
    ue_dict = dev_network.ue_dict
    for _id, ue in ue_dict.items():
        assert ue.ue_id == env.UE_DICT[_id].ue_id
        assert ue.location == env.UE_DICT[_id].location
        assert ue.velocity == env.UE_DICT[_id].velocity


def test_state_dim(dev_network):
    assert dev_network.state_dim == 7


def test_actions(dev_network):
    assert dev_network.n_actions == 2


def test_avg_app_sla(dev_network):
    assert dev_network.get_avg_app_sla(0, 0) == 0.0
    assert dev_network.get_avg_app_sla(10, 7) == 0.7


def test_get_ap_stats(dev_network):
    n_ues_dict, avg_sla_dict = dev_network.get_ap_stats(8)
    assert n_ues_dict["video"] == 0
    assert n_ues_dict["web"] == 10
    assert avg_sla_dict["video"] == 0
    assert avg_sla_dict["web"] == 1.0


def test_get_network_state(dev_network):
    assert dev_network.get_network_state(env.UE1, 8) == env.NETWORK_STATE_1_8
    assert dev_network.get_network_state(env.UE2, 10) == env.NETWORK_STATE_2_10


def test_handoff(dev_network):

    # UE1 from AP8 to AP4
    assert len(env.AP4.n_ues["web"]) == 16
    assert env.AP4.ues_meeting_sla["web"] == 16
    assert dev_network.get_avg_app_sla(
        16, 16) == 1.0
    ap4 = dev_network.ap_dict[4]
    assert len(ap4.n_ues["web"]) == 16
    assert ap4.ues_meeting_sla["web"] == 16
    assert dev_network.get_avg_app_sla(
        17, 16) == 0.9
    ue_dict = dev_network.ue_dict
    ue = ue_dict[1]
    next_state = dev_network.perform_handoff(1, 4, env.NETWORK_STATE_1_8)
    assert not ue.sla
    assert ue.ap == 4
    assert next_state == env.NETWORK_STATE_1_4


def test_ue_reward(dev_network):
    assert dev_network.reward_based_on_ue_state(
        1, env.NETWORK_STATE_1_8, env.NETWORK_STATE_1_4) == -1


def test_ap_reward(dev_network):
    assert dev_network.reward_based_on_ap_state(
        1, env.NETWORK_STATE_1_8, env.NETWORK_STATE_1_4) == -1.5


def test_get_reward(dev_network):
    # handoff
    action = 1
    # UE's old sla
    ue_sla = 1

    assert dev_network.get_reward(
        action,
        env.NETWORK_STATE_1_8,
        env.NETWORK_STATE_1_4,
        env.UE1, ue_sla) == -2.5

    assert dev_network.get_reward(
        action,
        env.NETWORK_STATE_2_10,
        env.NETWORK_STATE_2_11,
        env.UE2,
        0) == -1.25

    assert dev_network.get_reward(
        action,
        env.NETWORK_STATE_2_10,
        env.NETWORK_STATE_2_14,
        env.UE2,
        0) == -3.25

    assert dev_network.get_reward(
        action,
        env.NETWORK_STATE_2_10,
        env.NETWORK_STATE_2_15,
        env.UE2,
        0) == -3.25
