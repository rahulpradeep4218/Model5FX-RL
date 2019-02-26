
import numpy as np
import ForexUtils as FXU
import os.path as ospath
import datetime
# Keras ML Imports
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten
from keras.optimizers import Adam
from keras.optimizers import SGD

# Keras RL Agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Trader Environment
from ForexEnv import ForexEnv


def create_model(shape, n_actions):
    model = Sequential()
    #model.add(LSTM(64, input_shape=shape, return_sequences=True))
    #model.add(LSTM(64))
    print("Input shape to the model : ",shape)
    model.add(Flatten(input_shape=shape))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))
    model.add(Activation('linear'))

    return model


def main():
    ml_variables = FXU.getMLVariables()
    sqlEngine = FXU.getSQLEngine()
    actions_table_details = {'name':'metaactions', 'col':['Action', 'Time'], 'type':['VARCHAR(20)', 'datetime'], 'null': [False,False]}
    ### Clear the actions table

    FXU.execute_query_db("DELETE FROM metaactions", sqlEngine)
    env = ForexEnv(type="train", inputSymbol="EURUSD", show_trade=True)
    env_test = ForexEnv(type="test", inputSymbol="EURUSD", show_trade=True)

    n_actions = env.action_space.n
    print("Number of actions : ",n_actions)
    model = create_model(shape=env.observation_space.shape, n_actions=n_actions)
    print(model.summary())


    #### Configuring the agent
    memory = SequentialMemory(limit=100000, window_length=env.window_size)
    policy = EpsGreedyQPolicy()

    # enable the dueling network
    # you can specify the dueling_type to one of {'avg','max','naive'}
    dqn = DQNAgent(model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-4), metrics=['mae'])
    if ml_variables['LoadWeights'] != 'no':
        path = 'model\\' + ml_variables['LoadWeights'] + ".h5f"
        if ospath.isfile(path):
            print("Weights exist, so Going to load the weights")
            dqn.load_weights(path)
    max_reward = -1000000

    while True:

        # Train :
        dqn.fit(env, nb_steps=env.split_point, nb_max_episode_steps=60000, visualize=False, verbose=2)

        try:
            info = dqn.test(env_test, nb_episodes=1, visualize=False)
            #reward = info.history['episode_reward']
            reward = env_test.balance - env_test.starting_balance
            print("Total Profit : ", reward)
            if reward > int(max_reward) and int(reward) != 0:
                max_reward = int(reward)
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                #np.array([info.history]).dump('./info/duel_dqn_reward_{0}_{1}.info'.format(env_test.symbol, max_reward))
                dqn.save_weights('./model/duel_dqn_reward_{0}_{1}.h5f'.format(env_test.symbol, max_reward), overwrite=True)
            #print("Info of testing : ",info.history)
            FXU.execute_query_db("INSERT INTO reinforcetests(Symbol,StartingBalance,TotalProfit,Time) VALUES('{0}','{1}','{2}','{3}')".format(env_test.symbol, env_test.starting_balance, reward, now), sqlEngine)
            #n_buys, n_lostBuys, n_sells, n_lostSells, portfolio = info['buys'], info['lostBuys'], info['sells'], info['lostBuys']
            #np.array([info]).dump('./info/duel_dqn_{0}_weights_{1}LS_{2}_{3}.info'.format(env_test.symbol, portfolio, n_buys, n_sells))
        except KeyboardInterrupt:
            return

        ##### Saving weights after each fitting to resume afterwards ###############
        if ml_variables['LoadWeights'] != 'no':
            dqn.save_weights(filepath='model\\' + ml_variables['LoadWeights'] + ".h5f", overwrite=True)

if __name__ == '__main__':
    main()