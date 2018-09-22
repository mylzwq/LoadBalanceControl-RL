#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Defines apis for clients """

from loadbalanceRL.utils import exceptions


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Tuesday, March 6th 2018, 11:07:15 am'


class Base:
    """
    Parent class for all cellular clients
    """
    def __new__(cls, environment_config):
        """
        Method allows to create new helper class without having them to call
        Base class everytime
        """
        base = super(Base, cls).__new__(cls)
        base.environment_config = environment_config
        base.server = environment_config['SERVER']
        base.server_port = environment_config['SERVER_PORT']
        return base

    # Override these private methods
    def _get_num_ues(self):
        """
        Private method to retrieve num of UEs currently present in the
        network
        """
        raise exceptions.ClientMethodNotImplemented(
            "_get_num_ues is not implemented for this client!"
        )

    def _get_num_aps(self):
        """
        Private method to retrieve num of APs currently present in the
        network
        """
        raise exceptions.ClientMethodNotImplemented(
            "_get_num_aps is not implemented for this client!"
        )

    def _reset_state(self):
        """
        Private method that implements logic to retrieve initial state
        of the environment
        """
        raise exceptions.ClientMethodNotImplemented(
            "_get_initial_state is not implemented for this client!"
        )

    # Public methods
    @property
    def num_ues(self):
        """
        Public method to fetch number of UEs
        """
        return self._get_num_ues()

    @property
    def num_aps(self):
        """
        Public method to fetch number of APs
        """
        return self._get_num_aps()

    def reset_state(self):
        """
        Public method to fetch initial state
        """
        return self._reset_state()
