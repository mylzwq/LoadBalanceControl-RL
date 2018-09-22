#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Sample cellular client for testing """

from tests.sample_files import sample_cellular_network as network
from loadbalanceRL.lib.environment.cellular import client_template


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


class CellularDevClient(client_template.Base):
    def __init__(self, environment_config):
        self.environment_config = environment_config

    def get_ap_list(self):
        """
        Method to get list of APs
        """
        return network.AP_LIST

    def get_ue_list(self):
        """
        Method to get list of UEs
        """
        return network.UE_LIST

    def perform_handoff(self, ue_id, ap_id):
        """
        Method to perform a handoff
        """
        ue = network.UE_DICT[ue_id]
        old_ap = network.AP_DICT[ue.ap]
        new_ap = network.AP_DICT[ap_id]

        ue.ap = ap_id
        # update old AP's stat after handoff
        old_ap.n_ues[ue.app].remove(ue_id)
        if ue.sla == 1:
            old_ap.ues_meeting_sla[ue.app] -= 1

        # update new AP's stat after handoff
        new_ap.n_ues[ue.app].add(ue_id)

        # UE's sla dropped after handoff
        ue.sla = 0

        return {'DONE': True,
                'UE': ue.to_dict,
                'OLD_AP': old_ap.to_dict,
                'NEW_AP': new_ap.to_dict}
