
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Collecting data for plotly the movement of the UES  """

__author__ = 'Mingyang Liu (liux3941@umn.edu)'
__date__ = 'Monday, July 30th 2018, 11:00:00 am'

def Info_plot_dynamic_UE(result, episode):
    """
    provide the information for plot of each episode

    Args:
        result: (from the result of run episode)

    Returns:
        The plot info in each episode for plot the UE and AP distribution
    """

    ue_ap_list = result.UE_AP_LIST[episode]
    nodes = [(0, 0)]
    nodes += [tuple(x.location) for x in ue_ap_list]

    br_list = result.BR_LIST
    BR_list_nodes = []
    # test_nodes += [ loci for x in br_list for loci in br_node(x.location)]
    BR_list_nodes += [loci for x in br_list for loci in br_node(x.location)]

    nodes += BR_list_nodes

    edges = []
    edge_color = []

    # edge_color = ['green'] * (len(edges))

    # add edges between UE and AP
    for i, ue in enumerate(ue_ap_list[16:]):
        if ue.sla == 1:
            color = "green"
        else:
            color = "red"
        edges.append((ue.ap, i+17))
        edge_color.append(color)

    ue_ap_edge = len(edges)

    nodes_color = []
    nodes_color += ['orange'] * 16

    for ue in ue_ap_list[16:]:
        if ue.app == 2:
            # video UEs
            color = 'blue'
        else:
            # Web UEs
            color = 'magenta'
        nodes_color.append(color)

    nodes_color += ['black'] * len(BR_list_nodes)

    n_x, n_y = zip(*nodes[1:])
    nodes_dict = dict(
        type='scatter',
        x=n_x,
        y=n_y,
        mode='markers',
        marker=dict(size=8, color=nodes_color)
    )

    edges_list = []
    for k, e in enumerate(edges):
        info = dict(
            type='scatter',
            x=[nodes[e[0]][0], nodes[e[1]][0]],
            y=[nodes[e[0]][1], nodes[e[1]][1]],
            mode='lines',
            line=dict(width=2, color=edge_color[k])
        )
        edges_list.append(info)

    # Including the BR info to nodes and edges

    i = 1
    step = ue_ap_edge
    for k in range(len(br_list)):
        edges.append((i+step, i+1+step))
        edges.append((i+1+step, i+2+step))
        edges.append((i+2+step, i+3+step))
        edges.append((i+step, i+3+step))
        i += 4

    # print(edges[ue_ap_edge:])

    edge_color += ['black'] * len(BR_list_nodes)

    for j, br_edge in enumerate(edges[len(ue_ap_list)-16:]):
        b = (br_edge[0] + 16, br_edge[1] + 16)
        info = dict(
            type='scatter',
            x=[nodes[b[0]][0], nodes[b[1]][0]],
            y=[nodes[b[0]][1], nodes[b[1]][1]],
            mode='lines',
            line=dict(width=2, color=edge_color[j+len(ue_ap_list)-16])
        )
        edges_list.append(info)

    data = edges_list + [nodes_dict]
    episode_data = dict(data=data)
    return episode_data


def plot_dynamic_data(result, Rec_dict, EPISODES):
    """
    Args
        ----
            result: result from the simulation
            Rec_dict: data for the plot of the road and building 
            using rectangle
            EPISODES: the total number of episodes.

        Returns
        -------
            data: the first frame data.
            Frames_data : data for each frame
            last_episode: data for last episode
            layout_dict: layout for plotly

    """
    Frames_data = []
    data = Info_plot_dynamic_UE(result, 0)

    for episode in range(1, EPISODES-1):
        episode_data = Info_plot_dynamic_UE(result, episode)
        Frames_data.append(episode_data)

    layout_dict = dict(
        xaxis={'range': [0, 1600], 'title': 'x'},
        yaxis={'range': [0, 1600], 'title': 'y'},
        title="UE and AP Distribution in the grid",
        height=600,
        width=800,
        showlegend=False,
        shapes=Rec_dict,
        )
#     layout_dict = dict(xaxis={'range': [0, 1600], 'title': 'x'},
#               yaxis={'range': [0, 1600], 'title': 'y'},
#               title="UE and AP Distribution in the grid",
#               height=800,
#               width=1000,
#               showlegend=False
#               )
    # layout = go.Layout(layout_dict)

    last_episode = Info_plot_dynamic_UE(result, EPISODES-1)
    layout = {'title': "UE and AP Distribution in the grid"}
    last_episode["layout"] = layout

    Frames_data.append(last_episode)
    return data, Frames_data, last_episode, layout_dict


def br_node(br_location):
    """
    return the location for each of the vertex
    """
    loc1 = (br_location[0], br_location[2])
    loc2 = (br_location[0], br_location[3])
    loc3 = (br_location[1], br_location[3])
    loc4 = (br_location[1], br_location[2])
    return [loc1, loc2, loc3, loc4]
