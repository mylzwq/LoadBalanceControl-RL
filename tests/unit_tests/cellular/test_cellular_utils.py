#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Test cases for cellular environment utilities """

from loadbalanceRL.lib.environment.cellular.dev import utils


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Friday, March 16th 2018, 3:48:24 pm'


AP_LIST = list(range(100, 900, 200))
UE_LOCATION = (345, 567)

# Create a sample UE and sample AP for testing
SAMPLE_CURRENT_AP = utils.AP(
    ap_id=11,
    location=(500, 500),
    n_ues={'web': {97, 5, 7, 170, 77, 180, 151, 121, 155},
           'video': {
               32, 1, 2, 161, 71, 103, 168, 75, 44, 112, 17, 51, 20, 55},
           'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 9, 'video': 10, 'voice': 0, 'others': 0},
    )

# pylint: disable=W0612
AP6 = utils.AP(  # noqa
    ap_id=6,
    location=(300, 300),
    n_ues={'web': {194, 131, 164, 10, 188, 116, 84, 123, 60, 190},
           'video': {33, 129, 193, 36, 173, 175, 49, 19, 61, 154, 29, 63},
           'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 10, 'video': 7, 'voice': 0, 'others': 0},
)

AP7 = utils.AP(
    ap_id=7,
    location=(300, 500),
    n_ues={'web': {39, 43, 107, 139, 171, 147, 94, 25, 28, 157, 30},
           'video': {35, 197, 73, 140, 13},
           'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 11, 'video': 5, 'voice': 0, 'others': 0},
)

SAMPLE_UE = utils.UE(
    ue_id=1,
    ap=11,
    location=(488, 467),
    app='video',
    required_bandwidth=2.0,
    neighboring_aps=[6, 10, 7],
    distance=35.114,
    throughput=2.0,
    sla=1,
    signal_power=-3,
    velocity=0,
    direction="N",
    location_type="on_road",
    br_id=1,
)

SAMPLE_BR1 = utils.BR(
    br_id=1,
    br_type="road",
    location=(484, 492, 400, 600),
    direction="NS"
)

SAMPLE_BR2 = utils.BR(
    br_id=2,
    br_type="building",
    location=(1000, 1005, 1002, 1006),
    direction="random"
)

BR_DICT={1:SAMPLE_BR1, 2:SAMPLE_BR2}

def test_ap():
    """
    Tests if AP class is properly initialized
    """
    assert isinstance(AP6.to_dict, dict)
    assert AP6._initialize_n_ues() == {
        "web": set(), "video": set(), "voice": set(), "others": set()}
    assert AP7._initialize_ues_slas() == {
        "web": 0, "video": 0, "voice": 0, "others": 0}


def test_get_interval():
    """
    Tests get_interval function
    """
    assert utils.get_interval(345, AP_LIST) == (300, 500)


def test_valid_ap():
    """
    Tests valid_ap function
    """
    assert utils.valid_ap((500, 700), AP_LIST)
    assert not utils.valid_ap((400, 267), AP_LIST)


def test_aps_in_grid():
    """
    Tests get_aps_in_grid function
    """
    assert utils.get_aps_in_grid(UE_LOCATION, AP_LIST) ==\
        [(500, 500), (300, 700), (300, 500), (500, 700)]


def test_valid_neighbors():
    """
    Tests get_valid_neighbors
    """
    ap = (500, 700)
    assert utils.get_valid_neighbors(ap, AP_LIST) ==\
        [(300, 700), (700, 700), (500, 500)]


def test_extended_neighboring_aps():
    """
    Tests get_extendedn_neighboring_aps
    """
    closest_aps = [(500, 500), (300, 700), (300, 500), (500, 700)]
    aps_per_axis = AP_LIST
    radius = 2

    assert utils.get_extended_neighboring_aps(
        closest_aps, aps_per_axis, radius) ==\
        [(100, 700), (500, 300), (300, 500),
         (500, 100), (300, 700), (500, 500),
         (300, 300), (700, 700), (500, 700),
         (700, 500), (100, 300), (700, 300),
         (300, 100), (100, 500)]


def test_get_neighboring_aps():
    """
    Tests get_neighboring_aps function
    """
    # With radius 1
    neighboring_aps = utils.get_neighboring_aps(UE_LOCATION, AP_LIST)
    assert neighboring_aps.within_grid ==\
        [(500, 500), (300, 700), (300, 500), (500, 700)]
    assert neighboring_aps.rest == []

    # With radius 2
    neighboring_aps = utils.get_neighboring_aps(UE_LOCATION, AP_LIST, 2)
    assert neighboring_aps.within_grid ==\
        [(500, 500), (300, 700), (300, 500), (500, 700)]
    assert neighboring_aps.rest ==\
        [(100, 700), (500, 300), (300, 300), (700, 700),
         (700, 500), (100, 500)]


def test_ue_ap_distance():
    """
    Tests ue_ap_distance function
    """
    assert utils.get_ue_ap_distance(
        SAMPLE_UE.location, SAMPLE_CURRENT_AP.location) == 35.114

def test_ue_location_type():
    """
    Tests is_road_or_building function
    """
    location_type, velocity, ue_direction, br_id = utils.is_road_or_building(
        BR_DICT, SAMPLE_UE.location
    )
    assert location_type == "on_road"


def test_get_closest_ap_location():
    """
    Tests get_closest_ap_location function
    """
    assert utils.get_closest_ap_location(
        [(500, 500), (300, 700), (300, 500), (500, 700)],
        UE_LOCATION
    ) == (300, 500)


def test_get_center_grid():
    """
    Tests get_center_grid function
    """
    assert utils.get_center_grid(100, AP_LIST)[0] in range(250, 550)
    assert utils.get_center_grid(100, AP_LIST)[1] in range(250, 550)


def test_calculate_distance_factor():
    """
    Tests calculate_distance_factor function
    """
    assert utils.calculate_distance_factor(441.367, 100) == 0.11


def test_calculate_radio_bandwidth():
    """
    Tests calculate_radio_bandwidth function
    """
    assert utils.calculate_radio_bandwidth(0.11, 10.0) == 1.1


def test_calculate_network_bandwidth():
    """
    Tests calculate_network_bandwidth function
    """
    assert utils.calculate_network_bandwidth(58, 50.0) == 0.862


def test_get_ue_throughput():
    """
    Tests get_ue_throughput function
    """
    assert utils.get_ue_throughput(100, 441.367, 58, 50.0, 10.0, 0.25) == 0.25


def test_get_ue_sig_power():
    """
    Tests get_ue_sig_power function
    """
    assert utils.get_ue_sig_power(35.114) == -3


def test_get_ue_sla():
    """
    Tests get_ue_sla function
    """
    assert utils.get_ue_sla(2.0, 2.0)
    assert not utils.get_ue_sla(0.1, 0.25)


def test_main():
    """
    Tests for utils main function
    """
    assert utils.main()
