import Data_Process as FXDataProcess
import ForexUtils as FXUtils
import gym
from gym import  spaces
import numpy as np


# Position Constant :
LONG = 0
SHORT = 1
FLAT = 2

# Action Constant :
BUY = 0
SELL = 1
HOLD = 2


class ForexEnv(gym.Env):
    def __init__(self, show_trade=True, type="Train", inputSymbol):
        self.showTrade = show_trade
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.type = type
        ml_variables = FXUtils.getMLVariables()
        self.variables = ml_variables
        self.spread = float(ml_variables['Spread'])
        data = FXDataProcess.ProcessedData(inputSymbol,train_test_predict=type)
        data.addSimpleFeatures()
        data.apply_normalization()
        self.df = data.df.values
        self.rawData = data.rawData

        # Features
        self.n_features = self.df.shape[1]
        self.shape = (1, self.n_features + 4)

        # Action space and Observation space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(high=np.inf, low=-np.inf, shape=self.shape, dtype=np.float64)

        # Profit Calculation Variables
        self.OnePip = float(ml_variables['SymbolOnePip'])
        self.PipVal = float(ml_variables['SymbolPipValue'])
        self.MaxLossPerTrade = float(ml_variables['MaxLossPerTrade'])



    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}

        self.reward = 0

        if self.action == BUY:
            if self.position == FLAT:
                self.position = LONG
                self.entryPrice = self.rawData[self.current_tick]
                self.portfolio = self.balance + self.calculateProfit()

            elif self.position == SHORT:
                self.reward = self.calculateProfit()
                self.balance = self.balance + self.reward
                self.portfolio = self.balance



    def updateBalance(self):



    def calculateProfit(self):
        if self.position == LONG:
            profit = (((self.rawData[self.current_tick] - self.entryPrice) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
        elif self.position == SHORT:
            profit = (((self.entryPrice - self.rawData[self.current_tick]) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
        else:
            profit = 0

        return profit


    def calculateQuanity(self):
        return self.MaxLossPerTrade / (self.PipVal * float(self.variables['PipsDivision']))

    def reset(self):
        self.current_tick = 0
        print("Starting Episode : ",self.episode)

        # Positions
        self.buys = 0
        self.sells = 0


        # Clear the variables
        self.balance = float(self.variables['StartingBalance'])
        self.portfolio = float(self.balance)
        self.profit = 0

        self.action = HOLD
        self.position = FLAT
        self.done = False


