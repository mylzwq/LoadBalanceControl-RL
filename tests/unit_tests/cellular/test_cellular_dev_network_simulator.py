#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for dev network simulator """

import pytest
from tests.sample_files import sample_cellular_network as sim
from loadbalanceRL.lib.environment.cellular.dev import network


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


@pytest.fixture
def dev_network():
    """
    Create an instance of dev network simulator
    """
    return network.StaticNetwork(20, 16, 100, 1)


def test_ap(dev_network):
    """
    Tests AP creation
    """
    ap_2 = dev_network._ap_dict[2]
    assert ap_2.ap_id == 2
    assert ap_2.location[0] == 100
    assert ap_2.location[1] == 300

    ap_8 = dev_network._ap_dict[8]
    assert ap_8.ap_id == 8
    assert ap_8.location[0] == 300
    assert ap_8.location[1] == 700


def test_ue_throughput(dev_network):
    """
    Test UE throughput calculation
    """
    assert dev_network.calculate_ue_throughput(
        sim.AP8, sim.UE1.distance, sim.UE1.required_bandwidth
    ) == sim.UE1.throughput

    assert dev_network.calculate_ue_throughput(
        sim.AP10, sim.UE2.distance, sim.UE2.required_bandwidth
    ) == sim.UE2.throughput


def test_ue_sla(dev_network):
    """
    Test UE's sla calculation
    """
    before = dev_network.ue_sla_stats["Meets"]
    assert dev_network.calculate_ue_sla(
        sim.UE1.throughput, sim.UE1.required_bandwidth) == 1
    assert dev_network.ue_sla_stats["Meets"] == before + 1

    assert dev_network.calculate_ue_sla(
        sim.UE2.throughput, sim.UE2.required_bandwidth) == 0


def test_ue_signal_power(dev_network):
    """
    Test UE's signal power
    """
    assert dev_network.calculate_ue_signal_power(
        sim.UE1.distance) == sim.UE1.signal_power
    assert dev_network.calculate_ue_signal_power(
        sim.UE2.distance) == sim.UE2.signal_power


def test_neighboring_aps(dev_network):
    """
    Test UE's neighboring aps
    """
    assert dev_network.fetch_neighboring_aps(
        sim.UE1, sim.AP8) == sim.UE1.neighboring_aps
    assert dev_network.fetch_neighboring_aps(
        sim.UE2, sim.AP10) == sim.UE2.neighboring_aps


def test_handoff(dev_network):
    """
    Test handoff
    """
    handoff = dev_network.perform_handoff(
        sim.UE1.ue_id, sim.AP4.ap_id)
    assert handoff['DONE']
    ue_dict = handoff['UE']
    assert ue_dict['ap'] == sim.AP4.ap_id
