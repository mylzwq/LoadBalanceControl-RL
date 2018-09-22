#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Sample Cellular environment for testing """

from collections import OrderedDict
from loadbalanceRL.lib.environment.cellular import base
from loadbalanceRL.lib.environment import environment_template

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'

ACTIONS = {
    0: 'STAY',
    1: 'HANDOFF',
}

BR1=base.BR(
    br_id=1,
    br_type="road",
    location=(1,9,1,200),
    direction="NS"
)

UE1 = base.UE(
    ue_id=1,
    ap=8,
    location=(268, 776),
    neighboring_aps=[4],
    signal_power=-4,
    app=1,
    sla=1,
)

UE2 = base.UE(
    ue_id=2,
    ap=10,
    location=(512, 334),
    neighboring_aps=[11, 15, 14],
    signal_power=-3,
    app=2,
    sla=0,
)

AP4 = base.AP(
    ap_id=4,
    location=(100, 700),
    n_ues={
        'web': {161, 162, 258, 294, 265, 11, 204, 109, 110, 174, 244, 88,
                153, 188, 222, 94},
        'video': set(),
        'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 16, 'video': 0, 'voice': 0, 'others': 0}
)

AP8 = base.AP(
    ap_id=8,
    location=(300, 700),
    n_ues={
        'web': {1, 232, 299, 46, 15, 55, 215, 219, 30, 191},
        'video': set(),
        'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 10, 'video': 0, 'voice': 0, 'others': 0}
)

AP10 = base.AP(
    ap_id=10,
    location=(500, 300),
    n_ues={
        'web': {64, 65, 101, 6, 39, 104, 169, 264, 296, 114, 210, 116, 274,
                56, 57, 250, 62},
        'video': {12, 18, 279, 23, 151, 286, 287, 291, 41, 172, 203, 207, 86,
                  214, 97, 98, 225, 228, 108, 237, 121, 252, 253},
        'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 17, 'video': 5, 'voice': 0, 'others': 0}
)

AP11 = base.AP(
    ap_id=11,
    location=(500, 500),
    n_ues={
        'web': {128, 192, 67, 263, 136, 9, 105, 202, 266, 205, 269, 239, 295,
                178, 85, 152, 249, 282},
        'video': {4, 196, 71, 139, 140, 267, 78, 113, 82, 179, 20, 53, 243,
                  186, 187},
        'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 18, 'video': 7, 'voice': 0, 'others': 0}
)

AP14 = base.AP(
    ap_id=14,
    location=(700, 300),
    n_ues={
        'web': {159, 33, 66, 226, 100, 259, 70, 298, 43, 300, 254, 87, 60,
                190, 31},
        'video': set(),
        'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 15, 'video': 0, 'voice': 0, 'others': 0}
)

AP15 = base.AP(
    ap_id=15,
    location=(700, 500),
    n_ues={
        'web': {2, 35, 290, 38, 166, 167, 168, 297, 173, 251, 144, 273, 84,
                22, 118, 278, 90, 27},
        'video': set(),
        'voice': set(), 'others': set()},
    ues_meeting_sla={'web': 18, 'video': 0, 'voice': 0, 'others': 0}
)

AP_DICT = OrderedDict({4: AP4, 8: AP8, 10: AP10, 11: AP11, 14: AP14, 15: AP15})
UE_DICT = OrderedDict({1: UE1, 2: UE2})
BR_DICT=OrderedDict({1: BR1})

NETWORK_STATE_1_8 = base.NETWORK_STATE(
    ue_sla=1, app=1, sig_power=-4, video_ues=0, web_ues=10,
    avg_video_sla=0.0, avg_web_sla=1.0)

UE_AP_STATE_1_8 = base.UE_AP_STATE(
    app=1, sig_power=-4, video_ues=0, web_ues=10,
    avg_video_sla=0.0, avg_web_sla=1.0)

NETWORK_STATE_1_4 = base.NETWORK_STATE(
    ue_sla=0, app=1, sig_power=-4, video_ues=0, web_ues=17,
    avg_video_sla=0.0, avg_web_sla=0.9)

UE_AP_STATE_1_4 = base.UE_AP_STATE(
    app=1, sig_power=-4, video_ues=0, web_ues=17,
    avg_video_sla=0.0, avg_web_sla=0.9)

NETWORK_STATE_2_10 = base.NETWORK_STATE(
    ue_sla=0, app=2, sig_power=-3, video_ues=23, web_ues=17,
    avg_video_sla=0.2, avg_web_sla=1.0)

UE_AP_STATE_2_10 = base.UE_AP_STATE(
    app=2, sig_power=-3, video_ues=23, web_ues=17,
    avg_video_sla=0.2, avg_web_sla=1.0)


NETWORK_STATE_2_11 = base.NETWORK_STATE(
    ue_sla=0, app=2, sig_power=-3, video_ues=15, web_ues=18,
    avg_video_sla=0.5, avg_web_sla=1.0)

UE_AP_STATE_2_11 = base.UE_AP_STATE(
    app=2, sig_power=-3, video_ues=15, web_ues=18,
    avg_video_sla=0.5, avg_web_sla=1.0)


NETWORK_STATE_2_14 = base.NETWORK_STATE(
    ue_sla=0, app=2, sig_power=-3, video_ues=0, web_ues=15,
    avg_video_sla=0.0, avg_web_sla=1.0)

UE_AP_STATE_2_14 = base.UE_AP_STATE(
    app=2, sig_power=-3, video_ues=0, web_ues=15,
    avg_video_sla=0.0, avg_web_sla=1.0)


NETWORK_STATE_2_15 = base.NETWORK_STATE(
    ue_sla=0, app=2, sig_power=-3, video_ues=0, web_ues=18,
    avg_video_sla=0.0, avg_web_sla=1.0)

UE_AP_STATE_2_15 = base.UE_AP_STATE(
    app=2, sig_power=-3, video_ues=0, web_ues=18,
    avg_video_sla=0.0, avg_web_sla=1.0)


NETWORK_STATE_DICT = {
    (UE1, 8): NETWORK_STATE_1_8,
    (UE1, 4): NETWORK_STATE_1_4,
    (UE2, 10): NETWORK_STATE_2_10,
    (UE2, 11): NETWORK_STATE_2_11,
    (UE2, 14): NETWORK_STATE_2_14,
    (UE2, 15): NETWORK_STATE_2_15,
}

UE_AP_STATE_DICT = {
    (UE1, 8): UE_AP_STATE_1_8,
    (UE1, 4): UE_AP_STATE_1_4,
    (UE2, 10): UE_AP_STATE_2_10,
    (UE2, 11): UE_AP_STATE_2_11,
    (UE2, 14): UE_AP_STATE_2_14,
    (UE2, 15): UE_AP_STATE_2_15,
}


class SampleCellularEnv(environment_template.Base):
    """
    Sample Cellular environment
    """
    def __init__(self, env_config):
        """
        Initialize sample env
        """
        self.env_config = env_config
        self.ue_sla_stats = {"Meets": 1, "Doesnot": 1}

    @property
    def _actions(self):
        """
        Sample actions allowed in the env
        """
        return ACTIONS

    @property
    def _state_dim(self):
        """
        Sample state dim
        """
        return len(base.NETWORK_STATE_ATTRIBUTES)

    def _reset(self):
        """
        Sample action to reset the env
        """
        pass

    def get_network_state(self, ue, ap):
        """
        Simulates get_network_state of cellular base
        """
        return NETWORK_STATE_DICT[(ue, ap)]

    def get_ue_ap_state(self, ue, ap):
        """
        Simulates get_ue_ap_state of cellular base
        """
        return UE_AP_STATE_DICT[(ue, ap)]

    def _step(self, state, action, ue, next_ap):
        """
        Sample action to take a step
        """
        if state == NETWORK_STATE_1_8:
            return NETWORK_STATE_1_4, -2.5
        if state == NETWORK_STATE_2_10:
            return NETWORK_STATE_2_11, -1.25

    @property
    def ap_dict(self):
        """
        Sample AP dict
        """
        return AP_DICT

    @property
    def ue_dict(self):
        """
        Sample UE dict
        """
        return UE_DICT
    @property
    def br_dict(self):
        """
        Sample UE dict
        """
        return BR_DICT


