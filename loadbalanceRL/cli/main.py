#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Declares external APIs

e.g.:
(venv)$ loadbalanceRL --help
Using TensorFlow backend.
Rainman2's logging has been configured!
Usage: loadbalanceRL [OPTIONS] COMMAND [ARGS]...

 loadbalanceRL's cli

Options:
  --verbose BOOLEAN      show verbose output for debugging
  --epsilon_min FLOAT    min value for epsilon to stop updating
  --epsilon_decay FLOAT  rate at which epsilon gets updated
  --epsilon FLOAT        epsilon for epsilon-greedy policy
  --gamma FLOAT          discount factor
  --alpha FLOAT          learning rate
  --episodes INTEGER     numeber of episodes/epochs
  --learning_rate FLOAT  learning rate for gradient descent
  --help                 Show this message and exit.

Commands:
    Cellular  Arguments for cellular environment

Each Environment will list all the possible algorithms
(venv)$ loadbalanceRL --alpha 0.6 Cellular --help
Using TensorFlow backend.
loadbalanceRL's logging has been configured!
Usage: loadbalanceRL Cellular [OPTIONS] COMMAND [ARGS]...

  Arguments for cellular environment

Options:
  --env_type [Dev|Prod]  type of cellular network: Dev/Prod
  --help                       Show this message and exit.

Commands:
  qlearning_linear_regression  Qlearning with Linear Regressor as Function...
  qlearning_naive              Qlearning without any function approximator...
  qlearning_nn                 Qlearning with Neural Network as Function...
  qlearning_DQN                Qlearning with deep Q-network

For using qlearning with NN FA on stationary cellular env model:
loadbalanceRL --episodes 1000 --alpha 0.01 --gamma 0.9 --epsilon 0.01
Cellular qlearning_nn --l1_hidden_units 10 --l1_activation relu
"""

import logging
import click
from loadbalanceRL import RAINMAN3

__author__ = 'Ari Saha, Mingyang Liu'

RUNNING_ALG_CONFIG = RAINMAN3.algorithm_config
RUNNING_ENV_CONFIG = RAINMAN3.environment_config


@click.option('--episodes', type=click.INT,
              default=RUNNING_ALG_CONFIG['EPISODES'],
              help='numeber of episodes/epochs')
@click.option('--alpha', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['ALPHA'],
              help='learning rate')
@click.option('--gamma', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['GAMMA'],
              help='discount factor')
@click.option('--epsilon', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['EPSILON'],
              help='epsilon for epsilon-greedy policy')
@click.option('--epsilon_decay', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['EPSILON_DECAY'],
              help='rate at which epsilon gets updated')
@click.option('--epsilon_min', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['EPSILON_MIN'],
              help='min value for epsilon to stop updating')
@click.option('--learning_rate', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['LEARNING_RATE'],
              help='learning rate for the gradient method')
@click.option('--memory_size', type=click.INT,
              default=RUNNING_ALG_CONFIG['MEMORY_SIZE'],
              help='memory size for experience reply in DQN')
@click.option('--batch_size', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['BATCH_SIZE'],
              help='batch size for updating the evaluation net in DQN')
@click.option('--replace_target_iter', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['REPLACE_TARGET_ITER'],
              help='iteration steps to replace the params in the target net in DQN')
@click.option('--verbose', type=click.BOOL,
              default=RUNNING_ALG_CONFIG['VERBOSE'],
              help='show verbose output for debugging')

        
@click.group('cli')
def cli(episodes,
        alpha,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        learning_rate,
        memory_size,
        batch_size,
        replace_target_iter,
        verbose):
    # pylint: disable=too-many-arguments
    """
    Rainman2's cli
    """

    RUNNING_ALG_CONFIG['EPISODES'] = episodes
    RUNNING_ALG_CONFIG['ALPHA'] = alpha
    RUNNING_ALG_CONFIG['GAMMA'] = gamma
    RUNNING_ALG_CONFIG['EPSILON'] = epsilon
    RUNNING_ALG_CONFIG['EPSILON_DECAY'] = epsilon_decay
    RUNNING_ALG_CONFIG['EPSILON_MIN'] = epsilon_min
    RUNNING_ALG_CONFIG['LEARNING_RATE'] = learning_rate
    RUNNING_ALG_CONFIG['MEMORY_SIZE'] = memory_size
    RUNNING_ALG_CONFIG['BATCH_SIZE'] = batch_size
    RUNNING_ALG_CONFIG['REPLACE_TARGET_ITER'] = replace_target_iter
    RUNNING_ALG_CONFIG['VERBOSE'] = verbose


@cli.group('Cellular')
@click.option('--env_type', type=click.Choice(['Dev', 'Prod']),
              default='Dev',
              help='type of cellular network: Dev/Prod')
def Cellular(env_type):
    # pylint: disable=too-many-arguments
    """
    Arguments for cellular environment
    """

    RUNNING_ENV_CONFIG = RAINMAN3.update_env('Cellular')
    RUNNING_ENV_CONFIG['TYPE'] = env_type


@Cellular.command('qlearning_naive')
def qlearning_naive_cmd():
    """
    Qlearning without any function approximator
    Returns:
        Q: (dict)
            Q(s,a) values
        poliy: (object)
            optimal policy
    """
    logger = logging.getLogger(__name__)
    try:
        RAINMAN3.run_experiment('Cellular', 'Qlearning', 'Naive')
    except Exception as error:
        logger.exception(error)


@Cellular.command('qlearning_linear_regression')
def qlearning_linear_regression_cmd():
    # pylint: disable=invalid-name
    """
    Qlearning with Linear Regressor as Function Approximator
    Returns:
        Q: (dict)
            Q(s,a) values
        policy: (object)
            optimal policy
    """
    logger = logging.getLogger(__name__)
    try:
        RAINMAN3.run_experiment(
            'Cellular', 'Qlearning', 'LinearRegression')
    except Exception as error:
        logger.exception(error)


@Cellular.command('qlearning_nn')
@click.option('--l1_hidden_units', type=click.INT,
              default=RUNNING_ALG_CONFIG['L1_HIDDEN_UNITS'],
              help='hidden units for layer-1')
@click.option('--l2_hidden_units', type=click.INT,
              default=RUNNING_ALG_CONFIG['L2_HIDDEN_UNITS'],
              help='hidden units for layer-2')
@click.option('--l1_activation', type=click.Choice(['relu']),
              default=RUNNING_ALG_CONFIG['L1_ACTIVATION'],
              help='type of activation for layer-1')
@click.option('--l2_activation', type=click.Choice(['relu']),
              default=RUNNING_ALG_CONFIG['L2_ACTIVATION'],
              help='type of activation for layer-2')
@click.option('--loss_function', type=click.Choice(['mean_squared_error']),
              default=RUNNING_ALG_CONFIG['LOSS_FUNCTION'],
              help='loss function')
@click.option('--learning_rate', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['LEARNING_RATE'],
              help='leanring rate in gradient method')
@click.option('--optimizer', type=click.Choice(['Adam']),
              default=RUNNING_ALG_CONFIG['OPTIMIZER'],
              help='optimizer used in the last layer')
def qlearning_nn_cmd(l1_hidden_units,
                     l2_hidden_units,
                     l1_activation,
                     l2_activation,
                     loss_function,
                     learning_rate,
                     optimizer):
    """
    Qlearning with Neural Network as Function Approximator
    """
    RUNNING_ALG_CONFIG['L1_HIDDEN_UNITS'] = l1_hidden_units
    RUNNING_ALG_CONFIG['L2_HIDDEN_UNITS'] = l2_hidden_units
    RUNNING_ALG_CONFIG['L1_ACTIVATION'] = l1_activation
    RUNNING_ALG_CONFIG['L2_ACTIVATION'] = l2_activation
    RUNNING_ALG_CONFIG['LOSS_FUNCTION'] = loss_function
    RUNNING_ALG_CONFIG['LEARNING_RATE'] = learning_rate
    RUNNING_ALG_CONFIG['OPTIMIZER'] = optimizer
    logger = logging.getLogger(__name__)
    try:
        RAINMAN3.run_experiment('Cellular', 'Qlearning', 'NN')
    except Exception as error:
        logger.exception(error)


@Cellular.command('qlearning_dqn')
@click.option('--l1_hidden_units', type=click.INT,
              default=RUNNING_ALG_CONFIG['L1_HIDDEN_UNITS'],
              help='hidden units for layer-1')
@click.option('--l2_hidden_units', type=click.INT,
              default=RUNNING_ALG_CONFIG['L2_HIDDEN_UNITS'],
              help='hidden units for layer-2')
@click.option('--l1_activation', type=click.Choice(['relu']),
              default=RUNNING_ALG_CONFIG['L1_ACTIVATION'],
              help='type of activation for layer-1')
@click.option('--l2_activation', type=click.Choice(['relu']),
              default=RUNNING_ALG_CONFIG['L2_ACTIVATION'],
              help='type of activation for layer-2')
@click.option('--loss_function', type=click.Choice(['mean_squared_error']),
              default=RUNNING_ALG_CONFIG['LOSS_FUNCTION'],
              help='loss function')
@click.option('--learning_rate', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['LEARNING_RATE'],
              help='learning rate for gradient descent')
@click.option('--memory_size', type=click.INT,
              default=RUNNING_ALG_CONFIG['MEMORY_SIZE'],
              help='memory size for experience reply in DQN')
@click.option('--batch_size', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['BATCH_SIZE'],
              help='batch size for updating the evaluation net in DQN')
@click.option('--replace_target_iter', type=click.FLOAT,
              default=RUNNING_ALG_CONFIG['REPLACE_TARGET_ITER'],
              help='iteration steps to replace the params in the target net in DQN')
@click.option('--optimizer', type=click.Choice(['Adam']),
              default=RUNNING_ALG_CONFIG['OPTIMIZER'],
              help='optimizer used in the last layer')
def qlearning_dqn_cmd(l1_hidden_units,
                     l2_hidden_units,
                     l1_activation,
                     l2_activation,
                     loss_function,
                     learning_rate,
                     memory_size,
                     batch_size,
                     replace_target_iter,
                     optimizer):
    """
    Deep Q network
    """
    RUNNING_ALG_CONFIG['L1_HIDDEN_UNITS'] = l1_hidden_units
    RUNNING_ALG_CONFIG['L2_HIDDEN_UNITS'] = l2_hidden_units
    RUNNING_ALG_CONFIG['L1_ACTIVATION'] = l1_activation
    RUNNING_ALG_CONFIG['L2_ACTIVATION'] = l2_activation
    RUNNING_ALG_CONFIG['LOSS_FUNCTION'] = loss_function
    RUNNING_ALG_CONFIG['OPTIMIZER'] = optimizer
    RUNNING_ALG_CONFIG['LEARNING_RATE'] = learning_rate
    RUNNING_ALG_CONFIG['MEMORY_SIZE'] = memory_size
    RUNNING_ALG_CONFIG['BATCH_SIZE'] = batch_size
    RUNNING_ALG_CONFIG['REPLACE_TARGET_ITER'] = replace_target_iter
    logger = logging.getLogger(__name__)
    try:
        RAINMAN3.run_experiment('Cellular', 'Qlearning', 'DQN')
    except Exception as error:
        logger.exception(error)


