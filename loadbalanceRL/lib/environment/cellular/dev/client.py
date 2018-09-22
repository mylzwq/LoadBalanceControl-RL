#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Provides Rest apis for interaction with Development network simulator.
"""

import logging
import requests
from collections import OrderedDict
from loadbalanceRL.lib.environment.cellular import client_template
from loadbalanceRL.lib.environment.cellular.dev import apis
from loadbalanceRL.utils import exceptions

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'

# Disable urllib3 logging
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.CRITICAL)


class CellularDevClient(client_template.Base):
    def __init__(self, environment_config):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.url = self._get_url_str()

    def _get_url_str(self):
        """
        Helper method to form complete url string
        """
        return "http://" + self.server + ":" + self.server_port

    def _format_get_req(self, get_call):
        """
        formats GET request based on get_call
        """
        req_string = '{}' + get_call
        return req_string.format(self.url)

    def make_get_call(self, request_string):
        """
        Helper to make HTTP requets
        """
        self.logger.debug(
            "Making GET request: {}".format(request_string))
        try:
            request = requests.get(request_string)
        except requests.exceptions.ConnectionError as error:
            self.logger.error(
                "Server: {} is not running!".format(self.url))
            self.logger.error("Error: {}".format(error))
        else:
            return request

    def _parse_output(self, response):
        """
        Helper to parse response data
        """
        if not response:
            raise exceptions.ExternalServerError(
                "Got empty response from the Server!")
        return response.json()["output"]

    def _format_and_parse(self, api):
        """
        Builds URL and makes the respective API request
        """
        api_request_call = self._format_get_req(api)
        response = self.make_get_call(api_request_call)
        return self._parse_output(response)

    def get_num_ues(self):
        """
        Method to retrieve number of UEs present in the cellular network
        """
        return self._format_and_parse(apis.NUM_UES)

    def get_num_aps(self):
        """
        Method to retrieve number of APs present in the cellular network
        """
        return self._format_and_parse(apis.NUM_APS)

    def get_ap_list(self):
        """
        Method to fetch list of APs from the network
        """
        return self._format_and_parse(apis.AP_LIST)

    def get_ap_info(self, ap_id):
        """
        Method to fetch details about an AP from the network
        """
        return self._format_and_parse(
            apis.AP_INFO + str(ap_id))

    def get_ue_list(self):
        """
        Method to fetch list of UEs from the network
        """
        return self._format_and_parse(apis.UE_LIST)

    def get_ue_info(self, ue_id):
        """
        Method to fetch details about an UE from the network
        """
        return self._format_and_parse(
            apis.UE_INFO + str(ue_id))

    def get_br_list(self):
        """
        Method to fetch list of UEs from the network
        """
        return self._format_and_parse(apis.BR_LIST)

    def get_br_info(self, br_id):
        """
        Method to fetch details about an UE from the network
        """
        return self._format_and_parse(
            apis.BR_INFO + str(br_id))

    def reset_network(self):
        """
        Method to re-initialize the network.
        """
        return self._format_and_parse(apis.RESET_NETWORK)

    def reset_network_after_move(self):
        """
        Method to re-initialize the network after one step movement of the ues
        """
        return self._format_and_parse(apis.RESET_NETWORK_AFTER_MOVE)

    def get_neighboring_aps(self, ue_id):
        """
        Method to retrieve list of neighboring APs for the UE
        """
        return self._format_and_parse(
            apis.NEIGHBORING_APS + str(ue_id))

    def get_ue_throughput(self, ue_id):
        """
        Method to calculate throughput of the UE
        """
        return self._format_and_parse(
            apis.UE_THROUGHPUT + str(ue_id))

    def get_ue_sla(self, ue_id):
        """
        Method to calculate UE's sla
        """
        return self._format_and_parse(
            apis.UE_SLA + str(ue_id))

    def get_ue_signal_power(self, ue_id):
        """
        Method to calculate signal power between the UE and AP
        """
        return self._format_and_parse(
            apis.UE_SIGNAL_POWER + str(ue_id))

    def get_ap_slas(self, ap_id):
        """
        Method to calculate SLAs for each app with respect to the AP
        """
        return self._format_and_parse(
            apis.AP_SLAS + str(ap_id))

    def perform_handoff(self, ue_id, ap_id):
        """
        Method to perform a handoff for the UE to a new AP
        """
        return self._format_and_parse(
            apis.HANDOFF + str(ue_id) + '/' + str(ap_id)
        )


def main():
    """
    Test locally!
    """
    CELLULAR_MODEL_CONFIG = OrderedDict(
        NAME='Cellular',
        TYPE='Dev',
        SERVER='0.0.0.0',
        SERVER_PORT='8000',
        VERBOSE=True,
    )

    client = CellularDevClient(CELLULAR_MODEL_CONFIG)
    try:
        ap = client.get_ap_info(0)
    except exceptions.ExternalServerError as error:
        print("Error: {}".format(error))
    else:
        print(ap)
    print("")

    try:
        ap = client.get_br_info(1)
    except exceptions.ExternalServerError as error:
        print("Error: {}".format(error))
    else:
        print(ap)
    print("")

    try:
        ue = client.get_ue_info(0)
    except exceptions.ExternalServerError as error:
        print("Error: {}".format(error))
    else:
        print(ue)
    print("")


if __name__ == '__main__':
    main()
