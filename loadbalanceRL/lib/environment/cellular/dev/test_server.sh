#!/bin/bash
#
# """ Shell script to start server and test """
# 
# 
# __author__ = 'Ari Saha (arisaha@icloud.com), Mingyang Liu(liux3941@umn.edu)'
# __date__ = 'Friday, March 9th 2018, 12:02:22 pm'
#

function start_server() {
    VAL=$(python -c 'import sys; print("1" if hasattr(sys, "base_prefix") else "0")')
    if [ $VAL != "0" ]
    then
        $(python server.py) &
    else
        printf 'virtualenv is not running!\n'
        exit
    fi
}

function test_index() {
    echo -e "Testing index"
    curl -i -H "Accept: application/json" -H "Content-Type: application/json" http://0.0.0.0:8000/index
    echo
    echo
}

function test_num_aps() {
    echo -e "Testing num_aps"
    curl -i -H "Accept: application/json" -H "Content-Type: application/json" http://0.0.0.0:8000/env_params/num_aps
    echo
    echo
}

function test_num_ues() {
    echo -e "Testing num_ues"
    curl -i -H "Accept: application/json" -H "Content-Type: application/json" http://0.0.0.0:8000/env_params/num_ues
    echo
    echo
}

function test_ue_list() {
    echo -e "Testing ue_list"
    curl -i -H "Accept: application/json" -H "Content-Type: application/json" http://0.0.0.0:8000/ue_list
    echo
    echo
}

function test_ap_list() {
    echo -e "Testing ap_list"
    # test num_ues
    curl -i -H "Accept: application/json" -H "Content-Type: application/json" http://0.0.0.0:8000/ap_list
    echo
    echo
}

function test_br_list() {
    echo -e "Testing br_list"
    # test num_ues
    curl -i -H "Accept: application/json" -H "Content-Type: application/json" http://0.0.0.0:8000/br_list
    echo
    echo
}

function test_apis() {
    # Test index
    test_index

    # Test num_aps
    test_num_aps

    # Test num_ues
    test_num_ues

    # Test ap_list
    test_ap_list

    # Test ue_list
    test_ue_list
    
    # test br_lisy
    test_br_list
}

function stop_server() {
    pkill -f server.py
}

start_server
sleep 1
test_apis
stop_server