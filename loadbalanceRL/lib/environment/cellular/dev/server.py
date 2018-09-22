#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Defines external aps for Static Cellular model """


from flask import Flask, Response
import simplejson as json
import network
import apis

__author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'


# Network config
# Set DEV_NETWORK to Static or Dynamic
DEV_NETWORK = 'Static'
NUM_UES = 200
NUM_APS = 16
SCALE = 200.0
EXPLORE_RADIUS = 1
NUM_BRS = 4

# Server config
HOST = '0.0.0.0'
PORT = 8000
DEBUG = True

if DEV_NETWORK == 'Dynamic':
    network_model = network.DynamicNetwork
else:
    # Default is Static Network
    network_model = network.StaticNetwork
CELLULAR_NETWORK = network_model(
    NUM_UES, NUM_APS, SCALE, NUM_BRS, EXPLORE_RADIUS
    )
# CELLULAR_NETWORK.reset_network_after_move()

app = Flask(__name__)


class SetEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to deal with Sets
    """
    # pylint: disable=E0202
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def jsonify_params(value):
    """
    Helper method to jsonify response
    """
    data = {
        "output": value,
    }
    js = json.dumps(data, cls=SetEncoder)
    resp = Response(js, status=200, mimetype='application/json')
    return resp


@app.route('/')
@app.route(apis.INDEX, methods=['GET'])
def index():
    return jsonify_params(
        "Static Cellular network Simulator is ready!!"
    )


@app.route(apis.NUM_UES, methods=['GET'])
def get_num_ues():
    """
    Method to return num of ues present in the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.num_ues
    )


@app.route(apis.NUM_APS, methods=['GET'])
def get_num_aps():
    """
    Method to return num of aps present in the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.num_aps
    )


@app.route(apis.AP_LIST, methods=['GET'])
def get_ap_list():
    """
    Method to return list of aps present in the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.ap_list
    )


@app.route(apis.AP_INFO + '<ap_id>', methods=['GET'])
def get_ap_info(ap_id):
    """
    Method to return info about an AP from the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.ap_info(int(ap_id))
    )


@app.route(apis.UE_LIST, methods=['GET'])
def get_ue_list():
    """
    Method to return list of ues present in the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.ue_list
    )


@app.route(apis.UE_INFO + '<ue_id>', methods=['GET'])
def get_ue_info(ue_id):
    """
    Method to return info about an UE from the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.ue_info(int(ue_id))
    )


@app.route(apis.BR_LIST, methods=['GET'])
def get_br_list():
    """
    Method to return list of brs in the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.br_list
    )


@app.route(apis.BR_INFO + '<br_id>', methods=['GET'])
def get_br_info(br_id):
    """
    Method to return info about an BR from the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.br_info(int(br_id))
    )


@app.route(apis.RESET_NETWORK, methods=['GET'])
def reset_network():
    """
    Method to return initial state of the network
    """
    return jsonify_params(
        CELLULAR_NETWORK.reset_network()
    )


@app.route(apis.RESET_NETWORK_AFTER_MOVE, methods=['GET'])
def reset_network_after_move():
    """
    Method to return state of the network after move
    """
    return jsonify_params(
        CELLULAR_NETWORK.reset_network_after_move()
    )


@app.route(apis.NEIGHBORING_APS + '<ue_id>', methods=['GET'])
def neighboring_aps(ue_id):
    """
    Method to retrieve list of neighboring APs for the UE
    """
    return jsonify_params(
        CELLULAR_NETWORK.ue_neighboring_aps(int(ue_id))
    )


@app.route(apis.UE_THROUGHPUT + '<ue_id>', methods=['GET'])
def ue_throughput(ue_id):
    """
    Method to calculate throughput of the UE
    """
    return jsonify_params(
        CELLULAR_NETWORK.ue_throughput(int(ue_id))
    )


@app.route(apis.UE_SLA + '<ue_id>', methods=['GET'])
def ue_sla(ue_id):
    """
    Method to calculate UE's SLA
    """
    return jsonify_params(
        CELLULAR_NETWORK.ue_sla(int(ue_id))
    )


@app.route(apis.UE_SIGNAL_POWER + '<ue_id>', methods=['GET'])
def ue_signal_power(ue_id):
    """
    Method to calculate signal power between the UE and AP
    """
    return jsonify_params(
        CELLULAR_NETWORK.ue_signal_power(int(ue_id))
    )


@app.route(apis.AP_SLAS + '<ap_id>', methods=['GET'])
def ap_slas(ap_id):
    """
    Method to calculate all the APPs' SLAs for an AP
    """
    return jsonify_params(
        CELLULAR_NETWORK.ap_sla(int(ap_id))
    )


@app.route(apis.HANDOFF + '<ue_id>' + '/' + '<ap_id>', methods=['GET'])
def handoff(ue_id, ap_id):
    """
    Method to handoff UE to the new AP
    """
    return jsonify_params(
        CELLULAR_NETWORK.perform_handoff(int(ue_id), int(ap_id))
    )


def main():
    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == '__main__':
    main()
