#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Implementation of Qlearning for Cellular environments """

import logging
import numpy as np
import progressbar
from collections import namedtuple, defaultdict
import copy

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
__date__ = 'Wednesday, April 4th 2018, 11:13:00 am'

CELLULAR_RESULTS = namedtuple(
    'CELLULAR_RESULTS',
    ['Q', 'Q_ap', 'Rewards', 'Handoffs', 'Meets_SLA', 'UE_AP_LIST', 'BR_LIST'])

ACTION = namedtuple(
    'ACTION', ('action', 'ap_id', 'ue_ap_state'))

AP_INFO = namedtuple('AP_INFO', ('ap_id', 'ue_ap_state'))


class QlearningForCellular:
    # pylint: disable=invalid-name
    # pylint: disable=too-few-public-methods
    """
    Implements Qlearning algorithm for Cellular network
    """

    def __init__(self, algorithm_config, env, agent):
        """
        Declares local variables
        """
        self.algorithm_config = algorithm_config
        self.episodes = self.algorithm_config['EPISODES']
        self.env = env
        self.agent = agent
        self.episode = 0
        # setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

    def ap_with_max_q(self, ap_info_list):
        """
        Helper method to retrieve Q for each state and return ap with the
        highest Q
        """
        best_ap_info = ap_info_list[0]
        best_q = self.agent.q_from_ap_model(best_ap_info.ue_ap_state)

        for ap_info in ap_info_list[1:]:
            _q = self.agent.q_from_ap_model(ap_info.ue_ap_state)
            if _q > best_q:
                best_q = _q
                best_ap_info = ap_info
        return best_ap_info

    def get_ap_info(self, ue, ap_id):
        """
        Helper to create a new AP_INFO data structure for the given AP.

        Args
        ----
            ue: (UE object)
                Instance of UE.
            ap_id: (int)
                ID of AP.

        Returns
        -------
            ap_info: (AP_INFO object)
                Instance of AP_INFO containing ap_id and ue_ap_state
        """
        # Get UE_AP_STATE
        ue_ap_state = self.env.get_ue_ap_state(ue, ap_id)

        # Create a new AP_INFO and fill in the details.
        return AP_INFO(
            ap_id=ap_id,
            ue_ap_state=ue_ap_state)

    def get_next_best_ap(self, ue):
        """
        Method to retrieve Q value of corresponding states for each neighboring
        ap
        """
        neighboring_aps = ue.neighboring_aps
        ap_info_list = []
        for ap_id in neighboring_aps:
            ap_details = self.get_ap_info(ue, ap_id)
            ap_info_list.append(ap_details)
        return self.ap_with_max_q(ap_info_list)

    def get_next_action_with_neighbors(self, state, ue):
        """
        Method to take action when UE has neighboring APs
        """
        # Generate a random number between 0 and 1
        prob = np.random.rand()

        # List of neighboring aps, including current ap
        neighboring_aps_with_current = [ue.ap] + ue.neighboring_aps

        # Ask agent for next action
        cellular_agent_action = self.agent.take_action(
            state, neighboring_aps_with_current, prob)

        action = cellular_agent_action.action
        ap_id = cellular_agent_action.ap_id
        # Case when second Qtable is not utilized, such as, either action=Stay
        # or a random action in which case next ap is a random choice.
        ue_ap_state = None

        # Recommended action was 'Handoff' by the max Q(s,a). Must do a second Qtable lookup
        # to find the next best AP to handoff to.
        if action == -1:
            # ie. action is handoff, find the next best ap
            best_ap_info = self.get_next_best_ap(ue)
            ap_id = best_ap_info.ap_id
            ue_ap_state = best_ap_info.ue_ap_state

            # change action back to 1
            action = 1
        self.logger.debug(
            "Next action based on state is: {} and next_ap: {}".format(
                action, ap_id
            )
        )
        self.logger.debug(
            "ue_ap_state: {}".format(ue_ap_state)
        )

        return ACTION(action=action, ap_id=ap_id, ue_ap_state=ue_ap_state)

    def get_next_action(self, state, ue):
        """
        Method to take action based on the state
        """

        self.logger.debug(
            "Calculating action based on the state!")

        # Return action: Stay and UE's current ap if there are no neighboring
        # APs.
        if not ue.neighboring_aps:
            self.logger.debug(
                "There are no neighboring APs. Returning action=Stay")
            return ACTION(action=0, ap_id=ue.ap, ue_ap_state=None)
        return self.get_next_action_with_neighbors(state, ue)

    def calculate_env_sla(self):
        self.env.ue_sla_stats = defaultdict(int)
        for _id, ue in self.env.ue_dict.items():
            ue_sla = ue.sla
            if ue_sla == 1:
                self.env.ue_sla_stats["Meets"] += 1
            else:
                self.env.ue_sla_stats["Doesnot"] += 1

    def execute(self):
        """
        Main method to execute Qlearning algorithm
        """
        self.logger.info(
            "Running Qlearning for Cellular Networks!")

        # Per episode statistics
        # Rewards
        self.reward_stats = np.zeros(self.episodes)

        # Handoffs
        self.handoff_stats = np.zeros(self.episodes)

        # UEs meeting their SLAs
        self.ue_sla_stats = np.zeros(self.episodes)

        # UE and AP list for plotting
        self.ue_ap_list = [[False] for _ in range(self.episodes)]
        BR_LIST = []
        BR_LIST.extend(self.env.br_dict.values())

        # Track progress
        progress_bar = progressbar.ProgressBar(
            maxval=self.episodes,
            widgets=[progressbar.Bar('=', '[', ']'), ' ',
                     progressbar.Percentage()])
        progress_bar.start()

        # Keep generating experiences
        for episode in range(self.episodes):
            # get a starting state from the env
            # reset() will fetch latest list of UEs and APs and their stats.
            self.env.episode = episode
            if episode == 0:
                self.env.reset()
            else:
                self.env.reset_after_move()
            _ue_ap_list_per_episode = []
            _ue_ap_list_per_episode.extend(self.env.ap_dict.values())
            _ue_ap_list_per_episode.extend(self.env.ue_dict.values())

            self.ue_ap_list[episode] = copy.deepcopy(_ue_ap_list_per_episode)

            # self.ue_sla_stats[episode] = self.env.ue_sla_stats["Meets"]
            self.logger.debug("Running episode: {}".format(episode))

            for ue_id, ue in self.env.ue_dict.items():
                handoffs = 0
                self.logger.debug(
                    "#################  New UE: {} ###################".format(
                        ue_id))
                self.logger.debug(
                    "UE info: {}".format(ue.to_dict))
                self.logger.debug(
                    "UE's AP info: {}".format(
                        self.env.ap_dict[ue.ap].to_dict
                    )
                )

                # Get network's state
                state = self.env.get_network_state(ue, ue.ap)
                self.logger.debug(
                    "Starting state of the UE: {} is: {}".format(
                        ue_id, state))

                # Get next action based on current state and e-greedy policy.
                action, next_ap, ue_ap_state = self.get_next_action(
                    state, ue)
                if action == 1:
                    handoffs = 1

                # Take a step.
                # Find next state as a result of current action
                next_state, reward = self.env.step(state, action, ue, next_ap)
                self.logger.debug(
                    "Next_State based on action is: {}".format(
                        next_state))

                self.reward_stats[episode] += reward
                self.handoff_stats[episode] += handoffs

                # update agent's Q values
                self.agent.learn(
                    state, action, reward, next_state, ue_ap_state)

                # update progress_bar
                progress_bar.update(episode)

            self.calculate_env_sla()
            self.ue_sla_stats[episode] = self.env.ue_sla_stats["Meets"]

        self.logger.info("Episodes stats")
        self.logger.info(
            "Rewards: {}".format(self.reward_stats))
        self.logger.info(
            "Handoffs: {}".format(self.handoff_stats)
        )
        self.logger.info(
            "UEs SLA Stats: {}".format(self.ue_sla_stats)
        )
        # self.logger.info(
        #     "Total number of states encountered: {}".format(
        #         len(self.agent.Q))
        # )

        # Exit progress_bar
        progress_bar.finish()

        return CELLULAR_RESULTS(
            Q=self.agent.Q,
            Q_ap=self.agent.Q_ap,
            Rewards=self.reward_stats,
            Handoffs=self.handoff_stats,
            Meets_SLA=self.ue_sla_stats,
            UE_AP_LIST=self.ue_ap_list,
            BR_LIST=BR_LIST,
        )
