#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Sample Cellular Qagents for testing """

from tests.sample_files import sample_cellular_env as env
from loadbalanceRL.lib.algorithm.Qlearning.agents.tabular_q_learning import\
    CELLULAR_AGENT_ACTION


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


class SampleAgent:
    def __init__(self):
        self.model = {
            env.NETWORK_STATE_1_8: [-0.6, 0.0],
            env.NETWORK_STATE_2_10: [0.0, -0.25],
        }
        self.ap_model = {
            env.UE_AP_STATE_1_4: 1.0,
            env.UE_AP_STATE_2_11: 0.7,
            env.UE_AP_STATE_2_14: 0.4,
            env.UE_AP_STATE_2_15: 0.5,
        }

    def take_action(self, state, neighbors, prob):
        if state == env.NETWORK_STATE_1_8:
            return CELLULAR_AGENT_ACTION(-1, neighbors[0])
        if state == env.NETWORK_STATE_2_10:
            return CELLULAR_AGENT_ACTION(-1, neighbors[0])

    def q_from_ap_model(self, state):
        return self.ap_model[state]

    def learn(self, *args):
        pass

    @property
    def Q(self):
        return self.model

    @property
    def Q_ap(self):
        return self.ap_model
