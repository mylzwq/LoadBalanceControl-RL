# loadbalanceRL
Optimizing Dynamic Cellular network using Reinforcement Learning.
This work is based on the framework of loadbalanceRL
# Dependencies
* Python 3.6
# Authors
* Ari Saha  <arisaha@icloud.com> 
* Mingyang Liu  <liux3941@umn.edu>
  
Setup
=====
To install loadbalanceRL:

    $ virtualenv venv
    $ make install-prod
    $ source venv/bin/activate
    $ cp -r loadbalanceRL/etc rainman/venv/lib/python3.6/site-packages/loadbalanceRL-1.0-py3.6.egg/loadbalanceRL
    
  
Command Line
============
Here are the commands avaiable for loadbalanceRL.

    (venv)$ loadbalanceRL --help

        Using TensorFlow backend.
        Rainman2's logging has been configured!
        Usage: loadbalanceRL [OPTIONS] COMMAND [ARGS]...

            Rainman2's cli

        Options:
          --verbose BOOLEAN            show verbose output for debugging
          --replace_target_iter FLOAT  iteration steps to replace the params in the
                                       target net in DQN
          --batch_size FLOAT           batch size for updating the evaluation net in
                                       DQN
          --memory_size FLOAT          memory size for experience reply in DQN
          --learning_rate FLOAT        learning rate for the gradient method
          --epsilon_min FLOAT          min value for epsilon to stop updating
          --epsilon_decay FLOAT        rate at which epsilon gets updated
          --epsilon FLOAT              epsilon for epsilon-greedy policy
          --gamma FLOAT                discount factor
          --alpha FLOAT                learning rate
          --episodes INTEGER           numeber of episodes/epochs
          

        Commands:
            Cellular  Arguments for cellular environment
        Cellular Environment Experiments
================================

Cellular environments describe interactions of mobile devices (known as UEs) with cell towers (known as Access Points).

Rainman3 contains a simulated cellular environment called 'Dev' environment, which can be used to test various Reinforcement algorithms. Currently 'Dev' cellular environment supports the following algorithms:
     * Naive Qlearning, which is the generic form of Qlearning (Tabular).
     * Qlearning using linear function approximator, e.g. Linear regression.
     * Qlearning using non-linear function approximator, e.g. Neural Network.
     * Qlearning using Deep Q network, e.g. DQN.

To run experiments using Cellular network:
  1) If testing on development cellular network (i.e. --env_type = Dev), first start the development server (in a new terminal    tab), which will instantiate a simulated cellular network.


    (venv)$ cd loadbalanceRL/lib/environment/cellular/dev

    (venv)$ python server.py

2) To start running experiments on Cellular network, use the command line as below.


        (venv)$ loadbalanceRL Cellular --help

        Using TensorFlow backend.
        Rainman2's logging has been configured!
        Usage: loadbalanceRL Cellular [OPTIONS] COMMAND [ARGS]...

            Arguments for cellular environment

        Options:
            --env_type [Dev|Prod]  type of cellular network: Dev/Prod
            --help                 Show this message and exit.

        Commands:
            qlearning_linear_regression  Qlearning with Linear Regressor as Function...

            qlearning_naive              Qlearning without any function approximator...

            qlearning_nn                 Qlearning with Neural Network as Function...
            
            qlearning_dqn                DQN 



* With Tabular Q-learning algorithm


      (venv)$ loadbalanceRL --verbose True --episodes 50 Cellular --env_type Dev qlearning_naive


* With Linear regression Q-learning algorithm


      (venv)$ loadbalanceRL --verbose True --episodes 50 Cellular --env_type Dev qlearning_linear_regression


* With Neural network Q-learning algorithm

      (venv)$ loadbalanceRL --verbose True --episodes 50 Cellular --env_type Dev qlearning_nn

* With DQN

      (venv)$ loadbalanceRL --verbose True --episodes 50 Cellular --env_type Dev qlearning_dqn

Sample result plot for NN
================================
![alt text](https://github.com/mylzwq/LoadBalanceControl-RL/blob/master/loadbalanceRL/api/dynamic/Hanfoff.png)

![alt text](https://github.com/mylzwq/LoadBalanceControl-RL/blob/master/loadbalanceRL/api/dynamic/Rewards.png)

![alt text](https://github.com/mylzwq/LoadBalanceControl-RL/blob/master/loadbalanceRL/api/dynamic/UE_SLA.png)




Animation Plot
================================
(venv)$ jupyter notebook
* Ploat animation for dynamic environment with Q learning naive method

test_cellular_env.ipynb
* Ploat animation for dynamic environment with Q learning LinearRegression method

test_cellular_linear_dynamic.ipynb
* Ploat animation for dynamic environment with Q learning NN method

test_cellular_NN_dynamic.ipynb
* Ploat animation for dynamic environment with Q learning DQN method

test_cellular_dqn_dynamic.ipynb


