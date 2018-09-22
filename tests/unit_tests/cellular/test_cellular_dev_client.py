#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for cellular dev client """

import pytest
import requests
from collections import OrderedDict
from loadbalanceRL.utils import exceptions
from loadbalanceRL.lib.environment.cellular.dev import client

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


CELLULAR_DEV_CONFIG = OrderedDict(
    NAME='Cellular',
    TYPE='Dev',
    SERVER='0.0.0.0',
    SERVER_PORT='8000',
    VERBOSE=True,
)


@pytest.fixture
def client_instance():
    return client.CellularDevClient(CELLULAR_DEV_CONFIG)


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


def test_client_url(client_instance):
    assert client_instance._get_url_str() ==\
        "http://{}:{}".format(
            CELLULAR_DEV_CONFIG['SERVER'],
            CELLULAR_DEV_CONFIG['SERVER_PORT'])


@pytest.fixture
def test_url_str(client_instance):
    return client_instance._get_url_str()


def test_format_get_req(client_instance, test_url_str):
    api_call = '/num_aps'
    assert client_instance._format_get_req(api_call) ==\
        test_url_str + api_call


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_get_num_ues(client_instance):
    assert isinstance(client_instance.get_num_ues(), int)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_get_num_aps(client_instance):
    assert isinstance(client_instance.get_num_aps(), int)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ap_list(client_instance):
    assert isinstance(client_instance.get_ap_list(), list)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ue_list(client_instance):
    assert isinstance(client_instance.get_ue_list(), list)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ap_info_0(client_instance):
    with pytest.raises(
            exceptions.ExternalServerError,
            message="Not AP with id: 0"):
        client_instance.get_ap_info(0)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ap_info_1(client_instance):
    assert client_instance.get_ap_info(1)['ap_id'] == 1


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ue_info_0(client_instance):
    with pytest.raises(
            exceptions.ExternalServerError,
            message="Not UE with id: 0"):
        client_instance.get_ue_info(0)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ue_info_1(client_instance):
    assert client_instance.get_ue_info(1)['ue_id'] == 1


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ue_neighboring_aps(client_instance):
    print(client_instance.get_neighboring_aps(1))
    assert isinstance(client_instance.get_neighboring_aps(1), list)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ue_throughput(client_instance):
    assert isinstance(client_instance.get_ue_throughput(1), float)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ue_sla(client_instance):
    assert isinstance(client_instance.get_ue_sla(1), int)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ue_signal_power(client_instance):
    assert isinstance(client_instance.get_ue_signal_power(1), int)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_ap_slas(client_instance):
    assert isinstance(client_instance.get_ap_slas(2), dict)


@pytest.mark.skipif(not server(), reason="server is not running!")
def test_perform_handoff(client_instance):
    assert isinstance(client_instance.perform_handoff(1, 2), dict)
