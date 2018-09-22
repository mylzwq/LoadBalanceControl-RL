#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Utility for storing common lib and data structures """

import math
from collections import namedtuple
from itertools import product
import numpy as np
import simplejson as json
from numpy import random


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'

APPS_DICT = {"web": 0.25, "video": 2.0, "voice": 0.1, "others": 0.05}

NEIGHBORING_APS = namedtuple('NEIGHBORING_APS', ['within_grid', 'rest'])


class AP:
    def __init__(self,
                 ap_id=0,
                 location=None,
                 n_ues=None,
                 ues_meeting_sla=None,
                 max_connections=50,
                 uplink_bandwidth=25.0,
                 channel_bandwidth=10.0,
                 ):

        # Id of the AP
        self.ap_id = ap_id
        # location of the AP
        self.location = location
        # number of UEs currently connected to the AP
        # Dictionary with App_type as keys and list of ue_id as values
        self.n_ues = self._initialize_n_ues()
        # number of UEs meeting their SLAs
        self.ues_meeting_sla = self._initialize_ues_slas()
        # maximum connections AP can have
        self.max_connections = max_connections
        # uplink bandwidth of the AP
        self.uplink_bandwidth = uplink_bandwidth
        # channel bandwidth of the AP
        self.channel_bandwidth = channel_bandwidth

    def _initialize_n_ues(self):
        """
        Helper to setup an empty dictionary with type of Apps as keys.
        {"web": set(), "voice": set(), "video": set(), "others": set()}
        """
        return {key: set() for key in APPS_DICT.keys()}

    def _initialize_ues_slas(self):
        """
        Helper to setup an empty dictionary with type of Apps as keys.
        {"web": 0, "voice": 0, "video": 0, "others": 0}
        """
        return {key: 0 for key in APPS_DICT.keys()}

    @property
    def to_dict(self):
        """
        Formats class AP to a dict
        """
        return self.__dict__

    def __repr__(self):
        """
        Helper to represent AP in the form of:
        "AP {'ap_id: 4, 'location': (x, y), 'n_ues': 154}
        """
        return "<AP {}>".format(self.to_dict)


class UE:
    def __init__(self,
                 ue_id=0,
                 ap=0,
                 location=None,
                 app=None,
                 required_bandwidth=0,
                 neighboring_aps=None,
                 distance=0,
                 throughput=0,
                 sla=1,
                 signal_power=-100,
                 velocity=0,
                 direction=None,
                 location_type=None,
                 br_id=0,
                 ):

        # Id of the UE
        self.ue_id = ue_id
        # The access Point (AP) UE is conncted to. AP is identified by its
        # location
        self.ap = ap
        # Location of the UE (used for calculating sig_power)
        self.location = location
        # Type of application (Web/Video) the UE is running currently
        self.app = app
        # Required bandwidth for the UE based on the APP
        self.required_bandwidth = required_bandwidth
        # List of neighboring APs
        self.neighboring_aps = neighboring_aps
        # Distance between AP and UE
        self.distance = distance
        # UE's Throughput
        self.throughput = throughput
        # SLA. Default is meets SLA (1)
        self.sla = sla
        # Signal power between UE and AP
        self.signal_power = signal_power
        # velocity of the UE
        self.velocity = velocity
        # direction of the UE
        self.direction = direction
        # location type of the UE: in the build, on the road or else
        self.location_type = location_type
        # the id of the br related with the ue if the ue is in one building
        # or on one road
        self.br_id = br_id

    @property
    def to_dict(self):
        """
        Formats class UE to a dict
        """
        return self.__dict__

    @property
    def to_json(self):
        """
        Formats class UE to a json serializable format
        """
        return json.dumps(
            self, default=lambda o: o.to_dict, sort_keys=True, indent=4)

    def __repr__(self):
        """
        Helper to represent UE in the form of:
        "<UE {'ud_id': 1, 'location': (x, y), 'ap': 4}>
        """
        return "<UE {}>".format(self.to_dict)


class BR:
    def __init__(self, br_id=0, br_type=None, location=None, direction=None):
        """
        BR is the abbreviation for Building & Road
        """
        # Id of the BR
        self.br_id = br_id
        # type of the BR, it can be either "buidling" or "road"
        self.br_type = br_type
        # Location of the BR, in the form of (x_start, x_end, y_start, y_end)
        # the road and the building are both designed as rectangle
        self.location = location
        # direction of BR if is road, can be "NS" or "WE"
        # if the type is building, the direction is "random"
        self.direction = direction

    @property
    def to_dict(self):
        """
        Formats class BR to a dict
        """
        return self.__dict__

    @property
    def to_json(self):
        """
        Formats class UE to a json serializable format
        """
        return json.dumps(
            self, default=lambda o: o.to_dict, sort_keys=True, indent=4)

    def __repr__(self):
        """
        Helper to represent UE in the form of:
        "<BR {'BR_id': 1, 'type': 'road', location': (x1,x2,y1,y2),
        'direction':'NS'}>
        """
        return "<BR {}>".format(self.to_dict)


def get_br_type():
    """
    Function to randomly generate the type of road or building
    """
    prob = np.around(np.random.rand(), decimals=3)

    if prob < 0.7:
        return "road"
    # rest are "building"
    return "building"


def get_br_direction(br_type):
    """
    Function to randomly generate direction for UE along the
    road if the UE is on the road
    """
    if br_type == "road":
        prob = np.around(np.random.rand(), decimals=3)
        if prob < 0.5:
            return "NS"
        else:
            return "WE"
    else:
        return "random"


def get_ue_direction(br):
    """
    Function to randomly generate direction for
    UE along the road if the UE is on the road
    """
    br_direction = br.direction
    # br_type = br.br_type
    # br_direction = get_br_direction(br_type)
    if br_direction == "NS":
        prob = np.around(np.random.rand(), decimals=3)
        if prob < 0.5:
            return "N"
        else:
            return "S"
    elif br_direction == "WE":
        prob = np.around(np.random.rand(), decimals=3)
        if prob < 0.5:
            return "W"
        else:
            return "E"
    return "random"


def get_br_location(BR_type, scale, BR_direction, aps_per_axis):
    """
    Function to generate location for UR based on the UR type.

    The road will be placed in the "NS" or "WE" and the building will be
    located randomly in the range of less than 2 scale

    Args:
        app_type: (string):
        Type of road .

        scale: (float):
        Scale of each grid. e.g. 100.0 => Each grid is of 100.0 units

    Returns:
        location: (tuple)(x1,x2,y1,y2):
        Tuple of X start and X end and Y start and end in the grid.
    """
    max_grid = 1 + (2 * len(aps_per_axis)) * scale
    if BR_type == "road":
        width_road = 100
        if BR_direction == "NS":
            (x1, y1) = get_random_location(0, max_grid)
            x2 = min(x1+width_road, max_grid-1)
            y1 = 0
            y2 = max_grid-1
            # x2 = min(x1+width_road, max_grid)
            # y2 = np.random.randint(4*scale, max_grid)
            # y2 = max(6*scale, max_grid)
            (x1, x2) = sorted((x1, x2))
            (y1, y2) = sorted((y1, y2))
            return (x1, x2, y1, y2)
        elif BR_direction == "WE":
            (x1, y1) = get_random_location(0, max_grid)
            y2 = min(y1+width_road, max_grid-1)
            # x2 = np.random.randint(4*scale, max_grid)
            x1 = 0
            x2 = max_grid - 1
            # x2 = max(6*scale, max_grid)
            (x1, x2) = sorted((x1, x2))
            (y1, y2) = sorted((y1, y2))
            return (x1, x2, y1, y2)
    if BR_type == 'building':
        # if the building, we first generate the center point and
        # then the building is located with in a 20 units circle of the center
        (x1, y1) = get_random_location(0, max_grid)
        circle_r = 200
        x2_max = min(x1+circle_r, max_grid)
        x2_min = max(0, x1-circle_r)
        x2 = np.random.randint(x2_min, x2_max)
        y2_max = min(y1+circle_r, max_grid)
        y2_min = max(0, y1-circle_r)
        y2 = np.random.randint(y2_min, y2_max)
        (x1, x2) = sorted((x1, x2))
        (y1, y2) = sorted((y1, y2))
        return (x1, x2, y1, y2)


def change_direction_ue(ue_location, ue_direction, max_grid):
    (x, y) = ue_location
    if ue_direction == 'N':
        if y > max_grid-3:
            ue_direction = 'S'
    elif ue_direction == "S":
        if y < 2:
            ue_direction = "N"
    elif ue_direction == 'E':
        if x > max_grid-3:
            ue_direction = 'W'
    elif ue_direction == 'W':
        if x < 2:
            ue_direction = 'E'
    return ue_direction


def is_road_or_building(br_dict,
                        ue_location,
                        aps_per_axis,
                        scale,
                        ue_direction=None):
    """
    judge the ue is on the road or in the building or others
    and return the location type of the ue, the velocity of the ue
    and the ue's direction especially when it is on the road and
    the BR id the ue is conected to. if the ue is neither on the road
    nor  in the building the br_id is 0.
    """
    max_grid = (2 * len(aps_per_axis)) * scale
    for br_id, br in br_dict.items():
        br_type = br.br_type
        br_location = br.location
        if br_type == 'road' and is_on_road(br_location, ue_location, br):
            location_type = "on_road"
            if not ue_direction or ue_direction == "random":
                ue_direction = get_ue_direction(br)
            ue_direction = change_direction_ue(ue_location, ue_direction, max_grid)
            velocity = get_random_velocity(location_type)
            # located_br_id=br_id
            return [location_type, velocity, ue_direction, br_id]

        elif br_type == 'building' and is_in_buid(br_location, ue_location):
            location_type = "in_build"
            ue_direction = "random"
            velocity = get_random_velocity(location_type)
            # located_br_id=br_id
            return [location_type, velocity, ue_direction, br_id]

    location_type = "random_move"
    ue_direction = "random"
    velocity = get_random_velocity(location_type)
    br_id = 0

    return [location_type, velocity, ue_direction, br_id]


def is_in_buid(br_location, ue_location):
    """
    Args:
        ue_location:the location of the ue
                (x,y)tuple
    Returns:
            False/True: whether the ue is in the building
    """
    x_start = br_location[0]
    x_end = br_location[1]
    y_start = br_location[2]
    y_end = br_location[3]
    if ue_location[0] >= x_start and ue_location[0] <= x_end \
            and ue_location[1] >= y_start and ue_location[1] <= y_end:
        return True
    else:
        return False


def is_on_road(br_location, ue_location, br):
    """
        Args:
            br_location:
            (x1,x2,y1,y2) tuple
            ue_location:the location of the ue
                (x,y)tuple
            br:
            the BR information when it is a road
        Returns:
            False/True: whether the ue is on the road
    """
    x_start = br_location[0]
    x_end = br_location[1]
    y_start = br_location[2]
    y_end = br_location[3]

    br_direction = br.direction
    if br_direction == "NS":
        if ue_location[0] >= x_start and ue_location[0] <= x_end:
            return True
    elif br_direction == "WE":
        if ue_location[1] >= y_start and ue_location[1] <= y_end:
            return True
    else:
        return False


def get_random_velocity(location_type):
    if location_type == "on_road":
        velocity = int(random.randint(20, 60))
    elif location_type == "in_build":
        velocity = 1
        # velocity = random.random()
    else:
        velocity = 2
        # velocity = int(random.randint(0, 2))
    return velocity


def get_ue_location_after_move(br_info,
                               ue_location,
                               location_type,
                               velocity,
                               direction,
                               aps_per_axis,
                               scale):
    (x, y) = ue_location
    max_grid = (2 * len(aps_per_axis)) * scale
    if br_info:
        # br_location=br_info.location
        if location_type == 'on_road':
            if direction == 'N':
                # y = max_grid - 1
                y = min(y+velocity, max_grid - 1)
            elif direction == "S":
                # y=0
                y = max(y-velocity, 0)
            elif direction == "W":
                # x = 0
                x = max(x-velocity, 0)
            elif direction == "E":
                # x = max_grid - 1
                x = min(x+velocity, max_grid - 1)
        elif location_type == 'in_build':
            alpha = 2 * math.pi * random.random()
            x = velocity * math.cos(alpha) + x
            x = max(min(x, max_grid - 1), 0)
            # x=max(min(x,br_location[1]),br_location[0])
            y = velocity * math.sin(alpha) + y
            y = max(min(y, max_grid - 1), 0)
            # y=max(min(y,br_location[3]),br_location[2])
    else:
        alpha = 2 * math.pi * random.random()
        x = round(velocity * math.cos(alpha) + x, 0)
        x = max(min(x, max_grid - 1), 0)
        y = round(velocity * math.sin(alpha) + y, 0)
        y = max(min(y, max_grid - 1), 0)
    return (x, y)


def get_ue_app():
    """
    Function to randomly generate apps for UE and
    returns app_type and required bandwidth
    """
    prob = np.around(np.random.rand(), decimals=3)

    # 70% of UEs are running "web" application
    if prob < 0.7:
        return "web"
    # rest are running "video"
    return "video"


def get_random_location(_min, _max):
    """
    Function to generate random (x, y) between min and max
    """
    xloc = np.random.randint(_min, _max)
    yloc = np.random.randint(_min, _max)
    return (xloc, yloc)


def get_center_grid(scale, aps_per_axis):
    """
    Function to generate random x and y within 1.5*scale of radius
    """
    mid_point = sum(aps_per_axis) / len(aps_per_axis)
    _min = mid_point - 1.5*scale
    _max = mid_point + 1.5*scale
    return get_random_location(_min, _max)


def get_ue_location(app_type, scale, aps_per_axis):
    """
    Function to generate location for UE based on the app.

    UEs running video based apps will be placed in the center for the grid.
    This is designed so as to simulate 'high traffic load' in certain parts
    of the network which will force a handoff to neighboring APs.

    Args:
        app_type: (string):
        Type of application UE is running.

        scale: (float):
        Scale of each grid. e.g. 100.0 => Each grid is of 100.0 units

        aps_per_axis: (list):
        List of points in X-axis where APs are located.

    Returns:
        location: (tuple):
        Tuple of X and Y in the grid.
    """
    if app_type == "video":
        # place in within the center of the grid
        return get_center_grid(scale, aps_per_axis)
    # place it anywhere on the grid
    return get_random_location(0, (1 + (2 * len(aps_per_axis)) * scale))


def get_interval(value, num_list):
    """
    Helper to find the interval within which the value lies
    """
    if value < num_list[0]:
        return (num_list[0], num_list[0])
    if value > num_list[-1]:
        return (num_list[-1], num_list[-1])
    if value == num_list[0]:
        return (num_list[0], num_list[1])
    if value == num_list[-1]:
        return (num_list[-2], num_list[-1])

    for index, num in enumerate(num_list):
        if value <= num:
            return (num_list[index - 1], num_list[index])


def get_aps_in_grid(ue_location, aps_per_axis):
    """
    Function to retrieve a list of neighboring APs in the grid of the UE.
    """
    _min, _max = ue_location[0], ue_location[1]
    _min_interval = get_interval(_min, aps_per_axis)
    _max_interval = get_interval(_max, aps_per_axis)
    return list(set(product(_min_interval, _max_interval)))


def valid_ap(ap, aps_per_axis):
    """
    Helper to validate ap
    """
    ap_x, ap_y = ap
    return (ap_x in aps_per_axis and ap_y in aps_per_axis)


def get_valid_neighbors(ap, aps_per_axis):
    """
    Helper to return only valid neighbors
    """

    scale = aps_per_axis[1] - aps_per_axis[0]
    _aps = [
        (ap[0] - scale, ap[1]),
        (ap[0] + scale, ap[1]),
        (ap[0], ap[1] - scale),
        (ap[0], ap[1] + scale)]

    valid_aps = []

    for ap in _aps:
        if valid_ap(ap, aps_per_axis):
            valid_aps.append(ap)
    return valid_aps


def get_extended_neighboring_aps(closest_aps, aps_per_axis, radius):
    """
    Function to search for All APs within a given radius from the closest APs.
    """
    if not radius:
        return closest_aps

    all_aps = set(closest_aps)
    for ap in closest_aps:
        all_aps.update(get_valid_neighbors(ap, aps_per_axis))
    return get_extended_neighboring_aps(
        list(all_aps), aps_per_axis, radius - 1)


def get_neighboring_ap_ids(current_ap_location,
                           neighboring_aps,
                           location_to_ap_lookup):
    """
    Helper method to remove current ap from neighboring aps and return
    list of neighboring ap ids
    """
    neighboring_ap_ids = []
    if isinstance(current_ap_location, list):
        current_ap_location = tuple(current_ap_location)
    for ap_location in neighboring_aps:
        if current_ap_location != ap_location:
            neighboring_ap_ids.append(
                location_to_ap_lookup[ap_location])
    return neighboring_ap_ids


def update_neighboring_aps_after_move(ue, current_neighboring_aps_id, closest_ap_id):
    """
    update the neighboring APs for the UE after move, if the closest ap has changed,
    then assign the closest AP to the neighboring APS
    """
    # if the AP that the UE is currently connected has become the
    # neighbor APs by location, then remove it from the current
    # neighboring AP

    if ue.ap in current_neighboring_aps_id:
        current_neighboring_aps_id.remove(ue.ap)

    if closest_ap_id != ue.ap:
        current_neighboring_aps_id.insert(0, closest_ap_id)

    return current_neighboring_aps_id


def update_neighboring_aps(ue, new_ap, aps_per_axis, location_to_ap_lookup):
    """
        Method to update neighboring aps for the UE
    """
    neighboring_aps = get_neighboring_aps(
            ue.location, aps_per_axis)
    all_neighboring_aps =\
        neighboring_aps.within_grid + neighboring_aps.rest

    neighboring_ap_ids = get_neighboring_ap_ids(
        new_ap.location, all_neighboring_aps, location_to_ap_lookup
        )
    return neighboring_ap_ids


def get_aps_in_grid_inorder(ue_location, neighboring_aps_in_grid_unordered):
    """
    Function to retrieve a list of neighboring APs in the increasing order
    of ue_ap distance
    """
    def distance(p1, p2):
        return((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

    neighboring_aps_in_grid = sorted(
        neighboring_aps_in_grid_unordered,
        key=lambda p: distance(p, ue_location)
    )

    # neighboring_aps_in_grid = neighboring_aps_in_grid_unordered.sort(
    #     key = lambda p: (p[0] - x)**2 + (p[1]- y)**2
    # )
    return neighboring_aps_in_grid


def get_neighboring_aps(ue_location, aps_per_axis, radius=1):
    """
    Function to retrieve a list of neighboring APs with a given radius
    around the UE.
    """
    # these neighboring aps are distributed without order
    neighboring_aps_in_grid_unordered = get_aps_in_grid(ue_location, aps_per_axis)
    neighboring_aps_in_grid = get_aps_in_grid_inorder(ue_location, neighboring_aps_in_grid_unordered)
    rest = set()
    if radius > 1:
        rest.update(get_extended_neighboring_aps(
            neighboring_aps_in_grid, aps_per_axis, radius - 1))
        rest -= set(neighboring_aps_in_grid)
    return NEIGHBORING_APS(
        within_grid=neighboring_aps_in_grid, rest=list(rest))


def get_ue_ap_distance(ap_location, ue_location):
    """
    Function to calculate distance between UE and AP
    """
    ap_location = np.array(ap_location)
    ue_location = np.array(ue_location)
    return np.around(
        np.linalg.norm(ap_location - ue_location), decimals=3)


def get_closest_ap_location(neighboring_aps, ue_location):
    """
    Function that returns closest AP's location from the neighboring ap list
    """
    closest_ap = neighboring_aps[0]
    min_distance = get_ue_ap_distance(closest_ap, ue_location)
    for ap_location in neighboring_aps[1:]:
        distance = get_ue_ap_distance(ap_location, ue_location)
        if distance < min_distance:
            min_distance = distance
            closest_ap = ap_location
    return closest_ap


def get_ue_ap(ue_location, aps_per_axis, radius):
    """
    Function to retrive the closest AP to the UE
    """
    neighboring_aps = get_neighboring_aps(ue_location, aps_per_axis, radius)

    closest_ap_location = get_closest_ap_location(
        neighboring_aps.within_grid, ue_location)
    all_neighboring_aps = neighboring_aps.within_grid + neighboring_aps.rest
    return (closest_ap_location, all_neighboring_aps)


def calculate_distance_factor(ue_ap_distance, scale):
    """
    Function to calculate distance factor
    """
    return np.around(
        (math.exp(-(ue_ap_distance)/(2 * scale))), decimals=3)


def calculate_radio_bandwidth(distance_factor, ap_channel_bandwidth):
    """
    Function to calculate radio bandwidth of the AP
    """
    # calculate radio bandwidth
    return np.around((distance_factor * ap_channel_bandwidth), decimals=3)


def calculate_network_bandwidth(n_ues_on_ap, ap_uplink_bandwidth):
    """
    Function to calculate network bandwidth
    """
    # Ap factor
    ap_factor = 1
    # to avoid ZeroDivisionError
    if n_ues_on_ap:
        ap_factor /= n_ues_on_ap

    # network bandwidth
    return np.around((
        ap_factor * ap_uplink_bandwidth), decimals=3)


def get_ue_throughput(scale,
                      ue_ap_distance,
                      n_ues_on_ap,
                      ap_uplink_bandwidth,
                      ap_channel_bandwidth,
                      app_required_bandwidth):
    """
    Function to calculate throughput of UE
    """
    distance_factor = calculate_distance_factor(ue_ap_distance, scale)

    radio_bandwidth = calculate_radio_bandwidth(
        distance_factor, ap_channel_bandwidth)

    network_bandwidth = calculate_network_bandwidth(
        n_ues_on_ap, ap_uplink_bandwidth)

    return min(radio_bandwidth, network_bandwidth, app_required_bandwidth)


def get_ue_sig_power(ue_ap_distance):
    """
    Function to calculate signal power between the UE and AP
    """
    # To avoid ZeroDivisionError
    if ue_ap_distance:
        distance = (10 * math.log10(1 / math.pow(ue_ap_distance, 2)))
        # discretizing the distance
        distance /= 10
        return round(distance)


def get_ue_sla(ue_throughput, ue_required_bandwidth):
    """
    Function to calculate UE's SLA
    """
    return int(ue_throughput >= ue_required_bandwidth)


def main():
    """
    Test locally!
    """
    ap_list = list(range(100, 900, 200))
    print(ap_list)
    assert get_interval(345, ap_list) == (300, 500)
    ue_location = (345, 567)
    neighboring_aps = get_aps_in_grid(ue_location, ap_list)
    print(neighboring_aps)
    print(get_ue_ap_distance(neighboring_aps[0], ue_location))
    closest_ap = get_closest_ap_location(
        neighboring_aps, ue_location)
    print(closest_ap)

    (closest_ap, neighboring_aps) = get_ue_ap(ue_location, ap_list, 1)
    print(neighboring_aps)
    print(closest_ap)

    print("valid neighbors")
    print(get_valid_neighbors((500, 700), ap_list))

    print("Testing extended_neighboring_aps")
    print(get_extended_neighboring_aps(
        [(500, 500), (300, 700), (300, 500), (500, 700)], ap_list, 2))

    print("radius: 1")
    print(get_neighboring_aps(ue_location, ap_list, radius=1))
    print("radius: 2")
    print(get_neighboring_aps(ue_location, ap_list, radius=2))
    print("radius: 3")
    print(get_neighboring_aps(ue_location, ap_list, radius=3))
    print("radius: 4")
    print(get_neighboring_aps(ue_location, ap_list, radius=4))
    print("radius: 5")
    print(get_neighboring_aps(ue_location, ap_list, radius=5))
    print("radius: 6")
    print(get_neighboring_aps(ue_location, ap_list, radius=6))

    print(get_center_grid(100, ap_list))

    ue_ap_distance = 441.367
    assert calculate_distance_factor(ue_ap_distance, 100) == 0.11
    assert calculate_radio_bandwidth(0.11, 10.0) == 1.1
    assert calculate_network_bandwidth(58, 50.0) == 0.862
    assert get_ue_throughput(100, 441.367, 58, 50.0, 10.0, 0.25) == 0.25

    print(get_ue_sig_power(ue_ap_distance))
    return True


if __name__ == '__main__':
    main()
