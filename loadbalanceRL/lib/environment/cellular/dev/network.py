#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Simulates a cellular network for development/testing """

import logging
from logging.handlers import RotatingFileHandler
import math
from functools import reduce
from collections import defaultdict
from collections import OrderedDict
from loadbalanceRL.lib.environment.cellular.dev import utils

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


def setup_logging(module):
    logger = logging.getLogger(module)
    logger.setLevel(logging.DEBUG)
    handler = RotatingFileHandler(
        "network.log", maxBytes=1048576, backupCount=20)
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class StaticNetwork:
    """
    Simulation of a cellular network where APs are statically positioned in
    every cell of the grid.
    """
    # pylint: disable=E1101
    def __init__(self, num_ues, num_aps, scale, num_brs, explore_radius=1):
        self.num_ues = num_ues
        self.num_aps = num_aps
        self.scale = scale
        self.explore_radius = explore_radius
        self.num_brs = num_brs
        # setup logger
        self.logger = setup_logging(self.__class__.__name__)

        self.logger.info("Specifications of the simulated cellular network:")
        self.logger.info("Number of UES: {}".format(self.num_ues))
        self.logger.info("Number of APs: {}".format(self.num_aps))
        self.logger.info("Scale of each grid: {}".format(self.scale))
        self.logger.info(
            "Explore radius for the APs: {}".format(self.explore_radius))
        self.logger.info("Number of BRS: {}".format(self.num_brs))
        # number of aps in x axis
        self.x_units = int(math.sqrt(self.num_aps))
        self.logger.debug("Number of APs in x axis: {}".format(self.x_units))

        # position of aps in each axis
        self.aps_per_axis = [
            (1 + (2 * i)) * self.scale for i in range(self.x_units)]
        self.logger.debug(
            "Location of APs in both axis: {}".format(self.aps_per_axis))

        # Populate AP information
        # location_to_ap_lookup is a local dictionary used to for fast lookup
        # of AP id based on the location.
        self._location_to_ap_lookup = {}
        # Create a Dictionary containing details about the APs within the grid.
        # _ap_dict is used locally only for network functions.
        self._ap_dict = self._place_aps()

        # Populate BR information
        # List containing details about the BRs.
        self._br_dict = self._place_brs()

        # UE Stats keeps track of number of UEs for each type of app. Only for
        # troubleshooting
        self.ue_app_stats = defaultdict(int)
        self.ue_sla_stats = defaultdict(int)

        # List containing details about the UEs within the grid.
        self._ue_dict = self._instantiate_ues()
        self.logger.debug("Simulated Network summary:")
        self.logger.debug(
            "AP_DICT: {}".format(self._ap_dict))
        self.logger.debug(
            "UE_DICT: {}".format(self._ue_dict))
        self.logger.debug(
            "UE_SLA_STATS: {}".format(self.ue_sla_stats))

    """ Methods for fetching Neighboring APs """
    def neighboring_ap_ids(self, current_ap_location, neighboring_aps):
        """
        Helper method to remove current ap from neighboring aps and return
        list of neighboring ap ids
        """
        neighboring_ap_ids = []
        for ap_location in neighboring_aps:
            if current_ap_location != ap_location:
                neighboring_ap_ids.append(
                    self._location_to_ap_lookup[ap_location])
        return neighboring_ap_ids

    def fetch_neighboring_aps(self, ue, ap):
        """
        Method to fetch list of neighboring aps based on UE's location and
        current AP.
        """
        # Fetch list of neighboring aps
        self.logger.debug("Fetching neighboring AP list for the UE")
        neighboring_aps = utils.get_neighboring_aps(
            ue.location, self.aps_per_axis, self.explore_radius)
        all_neighboring_aps =\
            neighboring_aps.within_grid + neighboring_aps.rest
        neighboring_ap_ids = self.neighboring_ap_ids(
            ap.location, all_neighboring_aps
        )
        return neighboring_ap_ids

    def update_neighboring_aps(self, ue, new_ap):
        """
        Method to update neighboring aps for the UE
        """
        self.logger.debug("Updating UE: {}'s neighboring aps".format(
            ue.ue_id))
        ue.neighboring_aps = self.fetch_neighboring_aps(ue, new_ap)

    """ Method to calculate UE's stats """
    def calculate_ue_throughput(self,
                                current_ap,
                                ue_ap_distance,
                                ue_required_bandwidth):
        """
        Helper to calculate ue_throughput for the current AP
        """
        ap_n_ues = self.total_ues(current_ap.n_ues)
        ap_uplink_bandwidth = current_ap.uplink_bandwidth
        ap_channel_bandwidth = current_ap.channel_bandwidth

        return utils.get_ue_throughput(
            self.scale,
            ue_ap_distance,
            ap_n_ues,
            ap_uplink_bandwidth,
            ap_channel_bandwidth,
            ue_required_bandwidth)

    def calculate_ue_sla(self, ue_throughput, ue_required_bandwidth):
        """
        Helper to calculate UE's SLA

        Returns:
             1: meets
            -1: doen't meet
        """
        meets = utils.get_ue_sla(ue_throughput, ue_required_bandwidth)
        if not meets:
            self.ue_sla_stats["Doesnot"] += 1
        else:
            self.ue_sla_stats["Meets"] += 1
        return meets

    def calculate_ue_signal_power(self, ue_ap_distance):
        """
        Helper to calculate UE's signal power
        """
        return utils.get_ue_sig_power(ue_ap_distance)

    def update_ue_stats(self, ue, ap):
        """
        Helper method to update UE's stats
        """
        self.logger.debug("Updating UE's stats!")
        # Update UE-AP distance
        ue.distance = utils.get_ue_ap_distance(
            ue.location, ap.location
        )
        # Update UE's throughput
        ue.throughput = self.calculate_ue_throughput(
            ap, ue.distance, ue.required_bandwidth
        )
        # Update UE's SLA
        ue.sla = self.calculate_ue_sla(ue.throughput, ue.required_bandwidth)
        ap.ues_meeting_sla[ue.app] += ue.sla

        # Get new signal power based on UE-AP
        ue.signal_power = self.calculate_ue_signal_power(ue.distance)

    """ Methods to instantiating UEs and APs and BRs """
    def _place_aps(self):
        """
        Method to place APs in the grid.
        This method creates a dictionary of APs with location tuple (x, y) as
        key and AP obj as value.
        Each AP obj represents a row with ap_id, location, n_ues, etc.
        """
        self.logger.debug("Placing APs in respective grids")
        ap_dict = {}
        ap_id = 1
        # Get x-axis location
        for xloc in self.aps_per_axis:
            # Get y-axis location
            for yloc in self.aps_per_axis:
                # update location
                location = (xloc, yloc)
                self._location_to_ap_lookup[location] = ap_id

                ap_dict[ap_id] = utils.AP(
                    ap_id=ap_id, location=location)

                self.logger.debug("AP {} info:".format(ap_id))
                self.logger.debug(ap_dict[ap_id].to_dict)

                ap_id += 1
        self.logger.debug("APs have been successfully placed!")
        return ap_dict

    def total_ues(self, n_ues_dict):
        """
        Helper to sum total number of ues the AP has.
        """
        return reduce(
            lambda x, y: x+y,
            [len(values) for values in n_ues_dict.values()])

    def _place_brs(self):
        """
        Method to create BRs
        """
        self.logger.debug(
            "Instantiating {} BRs and placing them accordingly".format(
                self.num_brs))
        br_dict = {}

        for br_id in range(1, self.num_brs+1):
            br_type = utils.get_br_type()
            br_direction = utils.get_br_direction(br_type)
            br_location = utils.get_br_location(
                br_type, self.scale, br_direction, self.aps_per_axis
            )

            new_br = utils.BR(
                br_id=br_id,
                br_type=br_type,
                location=br_location,
                direction=br_direction
            )
            br_dict[br_id] = new_br
        return br_dict

    def _instantiate_ues(self):
        """
        Method to create UEs and connect them to their respective AP
        """
        self.logger.debug(
            "Instantiating {} UEs and placing them accordingly".format(
                self.num_ues))
        ue_dict = {}
        for ue_id in range(1, self.num_ues + 1):

            # Get app_type that the UE is running
            ue_app = utils.get_ue_app()
            self.ue_app_stats[ue_app] += 1
            required_bandwidth = utils.APPS_DICT[ue_app]

            # Get UE's location
            ue_location = utils.get_ue_location(
                ue_app, self.scale, self.aps_per_axis)

            # Get UE's closest AP
            (current_ap_location, neighboring_aps) = utils.get_ue_ap(
                ue_location, self.aps_per_axis, self.explore_radius)

            # Get current ap_id
            current_ap_id = self._location_to_ap_lookup[current_ap_location]

            # Update UE count for the AP
            current_ap = self._ap_dict[current_ap_id]
            current_ap.n_ues[ue_app].add(ue_id)

            # Update UE location type to be on the road or in the building
            [location_type, velocity, ue_direction, br_id] = utils.is_road_or_building(
                    self._br_dict, ue_location,
                    self.aps_per_axis, self.scale
                    )

            new_ue = utils.UE(
                ue_id=ue_id,
                ap=current_ap_id,
                location=ue_location,
                app=ue_app,
                required_bandwidth=required_bandwidth,
                neighboring_aps=self.neighboring_ap_ids(
                    current_ap_location, neighboring_aps
                ),
                location_type=location_type,
                velocity=velocity,
                direction=ue_direction,
                br_id=br_id
            )

            # Update new_ue's stats
            self.update_ue_stats(new_ue, current_ap)

            self.logger.debug("UE {} info:".format(ue_id))
            self.logger.debug(new_ue.to_dict)

            ue_dict[ue_id] = new_ue
        return ue_dict

    def _instantiate_ues_after_move(self):
        """
        Method to create new UEs after one move within the grid
        and connect them to their respective APs
        """

        for ue_id, ue in self._ue_dict.items():
            ue_id = ue_id
            ue_app = ue.app

            # Get update UE's location after one move
            ue_location_before = ue.location
            br_id = ue.br_id
            if br_id == 0:
                br_info = {}
            else:
                br_info = self._br_dict[br_id]
            ue_location_after_move = utils.get_ue_location_after_move(
                br_info, ue_location_before, ue.location_type, ue.velocity,
                ue.direction, self.aps_per_axis, self.scale
            )

            current_ap_id = ue.ap
            current_ap = self._ap_dict[current_ap_id]

            (closest_ap_location, current_neighboring_aps) = utils.get_ue_ap(
                ue_location_after_move, self.aps_per_axis, radius=1
                )

            closest_ap_id = self._location_to_ap_lookup[closest_ap_location]

            current_neighboring_aps_id = self.neighboring_ap_ids(
                closest_ap_location, current_neighboring_aps
                )
            neighboring_aps_id = utils.update_neighboring_aps_after_move(
                ue, current_neighboring_aps_id, closest_ap_id
                )

            # Get UE's closest AP
            # (current_ap_location, neighboring_aps) = utils.get_ue_ap(
            #     ue_location_after_move, self.aps_per_axis, radius=1)

            # neighboring_aps_id = self.neighboring_ap_ids(
            #     current_ap_location, neighboring_aps)

            # # Get current ap_id
            # current_ap_id = self._location_to_ap_lookup[current_ap_location]

            # # Update UE count for the AP
            # current_ap.n_ues[ue_app].add(ue_id)

            # Update UE location type to be on the road or in the building,
            # and the br_id related with the ue
            # if br_id=0, means none br is related with that ue
            [location_type, velocity, ue_direction, br_id] =\
                utils.is_road_or_building(
                    self._br_dict, ue_location_after_move,
                    self.aps_per_axis, self.scale, ue.direction
                )

            ue.location = ue_location_after_move
            ue.velocity = velocity
            ue.direction = ue_direction
            ue.ap = current_ap_id
            ue.location_type = location_type
            ue.neighboring_aps = neighboring_aps_id
            ue.distance = utils.get_ue_ap_distance(ue_location_after_move, current_ap.location)
            n_ues_on_ap = self.total_ues(current_ap.n_ues)

            ue_throughput = utils.get_ue_throughput(
                self.scale, ue.distance, n_ues_on_ap, current_ap.uplink_bandwidth,
                current_ap.channel_bandwidth, ue.required_bandwidth
            )

            ue.throughput = ue_throughput
            sla = self.calculate_ue_sla(
                ue.throughput, ue.required_bandwidth
            )
            if not sla and ue.sla:
                current_ap.ues_meeting_sla[ue_app] -= ue.sla
            if sla and not ue.sla:
                current_ap.ues_meeting_sla[ue_app] += ue.sla
            ue.sla = sla
            ue.signal_power = utils.get_ue_sig_power(ue.distance)

    def validate_ue(self, ue_id):
        """
        Helper method to validate if UE with ue_id exists
        """
        try:
            ue = self._ue_dict[ue_id]
        except KeyError as error:
            self.logger.exception(
                "UE with ue_id: {} doesn't exists! Error: {}".format(
                    ue_id, error))
            raise
        else:
            return ue

    def validate_br(self, br_id):
        """
        Helper method to validate if BR with br_id exists
        """
        try:
            br = self._br_dict[br_id]
        except KeyError as error:
            self.logger.exception(
                "BR with br_id: {} doesn't exists! Error: {}".format(
                    br_id, error))
            raise
        else:
            return br

    def validate_ap(self, ap_id):
        """
        Helper method to validate if AP with ap_id exists
        """
        try:
            ap = self._ap_dict[ap_id]
        except KeyError as error:
            self.logger.exception(
                "AP with ap_id: {} doesn't exists! Error: {}".format(
                    ap_id, error))
            raise
        else:
            return ap

    def handoff_to_ap(self, ue, current_ap, new_ap_id):
        """
        Method to handoff an UE to a new AP
        """
        self.logger.debug("Initiating Handoff!")

        # remove this ue from current AP
        self.logger.debug("Removing the UE from its current AP")
        current_ap.n_ues[ue.app].remove(ue.ue_id)
        current_ap.ues_meeting_sla[ue.app] -= ue.sla

        # locate the new AP
        new_ap = self.validate_ap(new_ap_id)

        self.logger.debug("Handing over the UE to new AP!")
        # update AP for the UE
        ue.ap = new_ap_id
        # add current UE to the requested AP
        new_ap.n_ues[ue.app].add(ue.ue_id)

        # update neighboring APs
        self.update_neighboring_aps(ue, new_ap)

        # update UE's stats
        self.update_ue_stats(ue, new_ap)

        self.logger.debug(
            "UE: {} is handed off from: {} to : {}".format(
                ue.ue_id, current_ap.ap_id, new_ap_id)
                )
        handoff_result = OrderedDict(
            DONE=True,
            UE=ue.to_dict,
            OLD_AP=current_ap.to_dict,
            NEW_AP=new_ap.to_dict
        )
        return handoff_result

    """ Internal APIs """

    def perform_handoff(self, ue_id, ap_id):
        """
        Method to simulate handoff of UE to a AP identified by its id
        """

        self.logger.debug(
            "Received request for a Handoff of UE: {} to AP: {}".format(
                ue_id, ap_id
            ))
        ue = self.validate_ue(ue_id)
        current_ap = self._ap_dict[ue.ap]

        self.logger.debug(
            "UE info: {}".format(ue.to_dict)
        )
        self.logger.debug(
            "UE's current_ap info: {}".format(current_ap.to_dict)
        )
        if ue.ap == ap_id:
            self.logger.debug(
                "Handoff: requested ap is same as current ap, aborting!")
            handoff_result = OrderedDict(
                DONE=False,
                UE=None,
                OLD_AP=None,
                NEW_AP=None
            )
            return handoff_result

        return self.handoff_to_ap(ue, current_ap, ap_id)

    def reset_network(self):
        """
        Re-initializes the network by instantiating APs and UEs again
        """
        self._ap_dict = None
        self._ue_dict = None

        # Place APs
        self._place_aps()

        # Instantiate UEs
        self._instantiate_ues()

    def reset_network_after_move(self):
        """
        Move the UE's direction and calculate the new state of the environment.
        The APs are placed as originally,
        but the UEs all moved according to the direction and velocity
        """
        # self._ap_dict = None
        # self._ue_dict = None
        # for ap_id in range(1, self.num_aps+1):
        #     ap = self._ap_dict[ap_id]
        #     ap.n_ues = {key: set() for key in utils.APPS_DICT.keys()}
        #     ap.ues_meeting_sla = {key: 0 for key in utils.APPS_DICT.keys()}

        # Instantiate UEs
        self._instantiate_ues_after_move()

    @property
    def ap_list(self):
        """
        Returns a list of APs.
        Converting ap_dict to ap_list to match what prod network api might
        send.
        """
        return [value.to_dict for value in self._ap_dict.values()]

    def ap_info(self, ap_id):
        """
        Method to return details about an AP
        """
        ap = self.validate_ap(ap_id)
        return ap.to_dict

    @property
    def br_list(self):
        """
        Returns a list of BRs.
        Converting br_dict to br_list
        """
        return [value.to_dict for value in self._br_dict.values()]

    def br_info(self, br_id):
        """
        Method to return details about an BR
        """
        br = self.validate_br(br_id)
        return br.to_dict

    @property
    def ue_list(self):
        """
        Returns a list of UEs
        Converting ue_dict to ue_list to match what prod network api might
        send.
        """
        return [value.to_dict for value in self._ue_dict.values()]

    def ue_info(self, ue_id):
        """
        Method to return details about an UE
        """
        ue = self.validate_ue(ue_id)
        return ue.to_dict

    def ue_throughput(self, ue_id):
        """
        Method to retrieve UE's throughput
        """
        ue = self.validate_ue(ue_id)
        return ue.throughput

    def ue_sla(self, ue_id):
        """
        Method to retrieve UE's sla
        """
        ue = self.validate_ue(ue_id)
        return ue.sla

    def ue_signal_power(self, ue_id):
        """
        Method to retrieve UE's signal_power
        """
        ue = self.validate_ue(ue_id)
        return ue.signal_power

    def ue_neighboring_aps(self, ue_id):
        """
        Method to reteive UE's neighboring aps
        """
        ue = self.validate_ue(ue_id)
        current_ap = self.validate_ap(ue.ap)
        return self.fetch_neighboring_aps(ue, current_ap)

    def ap_sla(self, ap_id):
        """
        Method to retrieve AP's sla
        """
        ap = self.validate_ap(ap_id)
        return ap.ues_meeting_sla


class DynamicNetwork:
    """
    Simulation of a cellular network where APs are dynamically positioned in
    every cell of the grid.
    """
    def __init__(self, num_ues, num_aps, scale):
        self.num_ues = num_ues
        self.num_aps = num_aps
        self.scale = scale


# TestNetwork=StaticNetwork(10, 16, 10)
def main():
    NUM_UES = 200
    NUM_APS = 16
    SCALE = 200.0
    EXPLORE_RADIUS = 1
    NUM_BRS = 2
    network_model = StaticNetwork
    CELLULAR_NETWORK = network_model(
        NUM_UES, NUM_APS, SCALE, NUM_BRS, EXPLORE_RADIUS)
    CELLULAR_NETWORK.perform_handoff(3, 1)
    CELLULAR_NETWORK.reset_network_after_move()


if __name__ == '__main__':
    main()
