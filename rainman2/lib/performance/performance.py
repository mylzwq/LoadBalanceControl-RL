
from loadbalanceRL.lib import interface
from collections import OrderedDict
from loadbalanceRL.settings import SETTINGS
    
ALGORITHM_CONFIG = OrderedDict(
        EPISODES=5,
        ALPHA=0.2,
        GAMMA=0.9,
        EPSILON=0.3,
        EPSILON_DECAY=0.99,
        EPSILON_MIN=0.01,
        VERBOSE=True,
        LEARNING_RATE=0.005,
        L1_HIDDEN_UNITS=13,
        L2_HIDDEN_UNITS=13,
        L1_ACTIVATION='relu',
        L2_ACTIVATION='relu',
        LOSS_FUNCTION='mean_squared_error',
        OPTIMIZER='Adam',
        REPLACE_TARGET_ITER=20,
        MEMORY_SIZE=2000,
        BATCH_SIZE=17,
    )

CELLULAR_MODEL_CONFIG = OrderedDict(
        NAME='Cellular',
        TYPE='Dev',
        SERVER='0.0.0.0',
        SERVER_PORT='8000',
        VERBOSE=True,
    )

def Performance(ALGORITHM_CONFIG, CELLULAR_MODEL_CONFIG, alog_name):
    """
    Performance testing
    """
    # Server profile: num_ues=200, APs=16, Scale=200.0, explore_radius=1
    loadbalanceRL = interface.Rainman2(SETTINGS)
    loadbalanceRL.algorithm_config = ALGORITHM_CONFIG
    loadbalanceRL.environment_config = CELLULAR_MODEL_CONFIG
    if alog_name=='linear':
        result_linear = loadbalanceRL.run_experiment(
           'Cellular', 'Qlearning', 'LinearRegression')
        return result_linear
    if alog_name=='Naive':
        result_Naive = loadbalanceRL.run_experiment(
           'Cellular', 'Qlearning', 'Naive')
        return result_Naive
    if alog_name=='NN':
        result_NN = loadbalanceRL.run_experiment(
           'Cellular', 'Qlearning', 'NN')
        return result_NN
    if alog_name=='DQN':
        result_DQN = loadbalanceRL.run_experiment(
           'Cellular', 'Qlearning', 'DQN')
        return result_DQN
    
if __name__ == '__main__':
    result_Naive = Performance(ALGORITHM_CONFIG, CELLULAR_MODEL_CONFIG, 'Naive')