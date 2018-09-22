#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Definition of all Rainman2 exceptions
"""


__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, February 14th 2018, 11:38:08 am'


class FileOpenError(IOError):
    """
    Exception raised when a file couldn't be opened.
    """
    pass


class AgentNotSupported(Exception):
    """
    Exception raised when agent is not valid for requested algorithm
    """
    pass


class AgentMethodNotImplemented(NotImplementedError):
    """
    Exception raised when trying to access a private method of an agent
    that is not implemented yet.
    """
    pass


class AlgorithmNotImplemented(NotImplementedError):
    """
    Exception raised when trying to access algorithm that is not
    implemented yet.
    """
    pass


class AlgorithmMethodNotImplemented(NotImplementedError):
    """
    Exception raised when trying to access a private method of an algorithm
    that is not implemented yet.
    """
    pass


class ClientNotImplemented(NotImplementedError):
    """
    Exception raised when trying to access client that is not
    implemented yet.
    """
    pass


class ClientMethodNotImplemented(NotImplementedError):
    """
    Exception raised when trying to access a private method of a client
    that is not implemented yet.
    """
    pass


class EnvironmentNotImplemented(NotImplementedError):
    """
    Exception raised when trying to access Environment that is not
    implemented yet.
    """
    pass


class EnvironmentMethodNotImplemented(NotImplementedError):
    """
    Exception raised when trying to access a private method of an environment
    that is not implemented yet.
    """
    pass


class ExternalServerError(Exception):
    """
    Exception raised when external server is not accessible
    """
    pass
