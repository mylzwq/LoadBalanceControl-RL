#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Sample Cellular network """

from loadbalanceRL.lib.environment.cellular.dev import utils

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'

# ***************************** UE1 ***************************** #
UE1 = utils.UE(
    ue_id=1,
    ap=8,
    location=(268, 776),
    app='web',
    required_bandwidth=0.25,
    neighboring_aps=[4],
    distance=82.462,
    throughput=0.25,
    sla=1,
    signal_power=-4
)


# ***************************** UE2 ***************************** #
UE2 = utils.UE(
    ue_id=2,
    ap=10,
    location=(512, 334),
    app='video',
    required_bandwidth=2.0,
    neighboring_aps=[11, 15, 14],
    distance=36.056,
    throughput=0.625,
    sla=0,
    signal_power=-3
)


# ***************************** AP4 ***************************** #
AP4 = utils.AP(
    ap_id=4,
    location=(100, 700),
)

AP4.n_ues = {
        'web': {161, 162, 258, 294, 265, 11, 204, 109, 110, 174, 244, 88,
                153, 188, 222, 94},
        'video': set(),
        'voice': set(), 'others': set()}
AP4.ues_meeting_sla = {'web': 16, 'video': 0, 'voice': 0, 'others': 0}


# ***************************** AP8 ***************************** #
AP8 = utils.AP(
    ap_id=8,
    location=(300, 700)
)

AP8.n_ues = {
        'web': {1, 232, 299, 46, 15, 55, 215, 219, 30, 191},
        'video': set(),
        'voice': set(), 'others': set()}
AP8.ues_meeting_sla = {'web': 10, 'video': 0, 'voice': 0, 'others': 0}


# ***************************** AP10 ***************************** #
AP10 = utils.AP(
    ap_id=10,
    location=(500, 300)
)

AP10.n_ues = {
        'web': {64, 65, 101, 6, 39, 104, 169, 264, 296, 114, 210, 116, 274,
                56, 57, 250, 62},
        'video': {12, 18, 279, 23, 151, 286, 287, 291, 41, 172, 203, 207, 86,
                  214, 97, 98, 225, 228, 108, 237, 121, 252, 253},
        'voice': set(), 'others': set()}
AP10.ues_meeting_sla = {'web': 17, 'video': 5, 'voice': 0, 'others': 0}


# ***************************** AP11 ***************************** #
AP11 = utils.AP(
    ap_id=11,
    location=(500, 500),
)
AP11.n_ues = {
        'web': {128, 192, 67, 263, 136, 9, 105, 202, 266, 205, 269, 239,
                295, 178, 85, 152, 249, 282},
        'video': {4, 196, 71, 139, 140, 267, 78, 113, 82, 179, 20, 53, 243,
                  186, 187},
        'voice': set(), 'others': set()}

AP11.ues_meeting_sla = {'web': 18, 'video': 7, 'voice': 0, 'others': 0}


# ***************************** AP14 ***************************** #
AP14 = utils.AP(
    ap_id=14,
    location=(700, 300)
)

AP14.n_ues = {
        'web': {159, 33, 66, 226, 100, 259, 70, 298, 43, 300, 254, 87, 60,
                190, 31},
        'video': set(),
        'voice': set(), 'others': set()}
AP14.ues_meeting_sla = {'web': 15, 'video': 0, 'voice': 0, 'others': 0}


# ***************************** AP15 ***************************** #
AP15 = utils.AP(
    ap_id=15,
    location=(700, 500)
)

AP15.n_ues = {
        'web': {2, 35, 290, 38, 166, 167, 168, 297, 173, 251, 144, 273, 84,
                22, 118, 278, 90, 27},
        'video': set(),
        'voice': set(), 'others': set()}
AP15.ues_meeting_sla = {'web': 18, 'video': 0, 'voice': 0, 'others': 0}


UE_LIST = [UE1.to_dict, UE2.to_dict]
UE_DICT = {UE1.ue_id: UE1, UE2.ue_id: UE2}

AP_LIST = [AP4.to_dict, AP8.to_dict,
           AP10.to_dict, AP11.to_dict,
           AP14.to_dict, AP15.to_dict]
AP_DICT = {AP4.ap_id: AP4,
           AP8.ap_id: AP8,
           AP10.ap_id: AP10,
           AP11.ap_id: AP11,
           AP14.ap_id: AP14,
           AP15.ap_id: AP15}
