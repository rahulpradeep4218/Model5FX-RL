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
    def __init__(self, inputSymbol, show_trade=True, type="train"):
        self.showTrade = show_trade
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.symbol = inputSymbol
        self.type = type
        ml_variables = FXUtils.getMLVariables()
        self.variables = ml_variables
        self.spread = float(ml_variables['Spread'])
        self.window_size = int(ml_variables['WindowSize'])

        data = FXDataProcess.ProcessedData(inputSymbol, train_test_predict=type)
        data.addSimpleFeatures()
        data.apply_normalization()
        self.df = data.df.values
        self.rawData = data.rawData
        self.split_point = data.split_point
        # Features
        self.n_features = self.df.shape[1]
        self.shape = (self.window_size, self.n_features + 3)

        # Action space and Observation space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(high=np.inf, low=-np.inf, shape=self.shape, dtype=np.float64)

        # Profit Calculation Variables
        self.OnePip = float(ml_variables['SymbolOnePip'])
        self.PipVal = float(ml_variables['SymbolPipValue'])
        self.MaxLossPerTrade = float(ml_variables['MaxLossPerTrade'])
        self.BoxPips = float(ml_variables['PipsDivision'])
        self.RiskRewardRatio = float(ml_variables['RiskRewardRatio'])



    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}

        #print("Action : ",action)
        self.action = action
        self.reward = 0.0
        self.current_price = self.getPrice(self.current_tick, 'C')

        if action == BUY:
            if self.position == FLAT:
                self.position = LONG
                self.entryPrice = self.getPrice(self.current_tick)
                self.stopLoss = self.entryPrice - (self.BoxPips * self.OnePip)
                self.takeProfit = self.entryPrice + (self.BoxPips * self.OnePip * self.RiskRewardRatio)
                self.portfolio = self.balance + self.calculateProfit()
                if self.type == "test":
                    FXUtils.execute_query_db("INSERT INTO metaactions(Action,Time) VALUES('BUY','" + str(self.rawData.index[self.current_tick]) + "')")

            elif self.position == SHORT and self.variables['UseTakeProfit'] != 'true':
                profit = self.calculateProfit()
                # self.reward += profit
                self.balance = self.balance + profit
                self.portfolio = self.balance
                self.position = FLAT
                self.sells += 1
                if profit < 0:
                    self.lostSells += 1


            else:
                self.updateBalance()

        elif action == SELL:
            if self.position == FLAT:
                self.position = SHORT
                self.entryPrice = self.rawData.iloc[self.current_tick]['Close']
                self.portfolio = self.balance + self.calculateProfit()
                self.stopLoss = self.entryPrice + (self.BoxPips * self.OnePip)
                self.takeProfit = self.entryPrice - (self.BoxPips * self.OnePip * self.RiskRewardRatio)
                if self.type == "test":
                    FXUtils.execute_query_db("INSERT INTO metaactions(Action,Time) VALUES('SELL','" + str(self.rawData.index[self.current_tick]) + "')")

            elif self.position == LONG and self.variables['UseTakeProfit'] != 'true':
                profit = self.calculateProfit()
                # self.reward += profit
                self.balance = self.balance + profit
                self.portfolio = self.balance
                self.position = FLAT
                self.buys += 1
                if profit < 0:
                    self.lostBuys += 1
            else:
                self.updateBalance()

        else:
            self.updateBalance()

        self.reward = self.calculate_reward()

        if self.showTrade and self.current_tick % 1000 == 0:
            print("Tick : {0} / Portfolio : {1} / Balance : {2}".format(self.current_tick, self.portfolio, self.balance))
            print("buys : {0} / {1} , sells : {2} / {3}".format((self.buys - self.lostBuys), self.lostBuys, (self.sells - self.lostSells), self.lostSells))
        self.history.append((self.action, self.current_tick, self.current_price, self.portfolio, self.reward))

        self.updateState()

        if self.current_tick > (self.df.shape[0] - self.window_size - 1) or self.portfolio < (0.25 * self.starting_balance):
            self.done = True
            self.reward += self.calculateProfit()
        self.current_tick += 1
        return self.state, self.reward, self.done, {'portfolio': np.array([self.portfolio]),
                                                    'buys': self.buys, 'lostBuys': self.lostBuys,
                                                    'sells': self.sells, 'lostSells': self.lostSells}



    def updateState(self):
        one_hot_position = FXUtils.getOneHotEncoding(self.position, 3)
        #profit = self.calculateProfit()
        #profit = self.portfolio - self.starting_balance
        # print("df : ", self.df[self.current_tick], " one hot : ",one_hot_position, " profit : ",[profit])
        self.state = np.concatenate([self.df[self.current_tick], one_hot_position])
        return self.state


    def updateBalance(self):
        if self.position == LONG:
            if self.getPrice(self.current_tick,type='L') <= self.stopLoss:
                profit = self.calculateProfit(type='S')
                # self.reward += profit
                self.balance = self.balance + profit
                self.portfolio = self.balance
                self.position = FLAT
                self.buys += 1
                self.lostBuys += 1

            elif self.getPrice(self.current_tick, type='H') >= self.takeProfit and self.variables['UseTakeProfit'] == 'true':
                profit = self.calculateProfit(type='T')
                # self.reward += profit
                self.balance = self.balance + profit
                self.portfolio = self.balance
                self.position = FLAT
                self.buys += 1


        elif self.position == SHORT:

            if self.getPrice(self.current_tick,type='H') >= self.stopLoss:
                profit = self.calculateProfit(type='H')
                # self.reward += profit
                self.balance = self.balance + profit
                self.portfolio = self.balance
                self.position = FLAT
                self.sells += 1
                self.lostSells += 1

            elif self.getPrice(self.current_tick, type='L') <= self.takeProfit and self.variables['UseTakeProfit'] == 'true':
                profit = self.calculateProfit(type='T')
                # self.reward += profit
                self.balance = self.balance + profit
                self.portfolio = self.balance
                self.position = FLAT
                self.sells += 1

    def calculate_reward(self):
        if self.position == FLAT:
            negStaticReward = float(self.variables['TotalStaticNegativeReward'])
            stepReward = (self.current_tick / self.df.shape[0]) * negStaticReward * 1.0
        else:
            stepReward = 0.0

        if self.max_portfolio == -10000:
            reward = 0.0
        else:
            reward = self.portfolio - self.max_portfolio
        if self.portfolio > self.max_portfolio:
            self.max_portfolio = self.portfolio
        return reward - stepReward

    def getPrice(self,index,type = 'C'):
        if type == 'O':
            return self.rawData.iloc[index]['Open']
        elif type == 'H':
            return self.rawData.iloc[index]['High']
        elif type == 'L':
            return self.rawData.iloc[index]['Low']
        elif type == 'C':
            return self.rawData.iloc[index]['Close']

    def calculateProfit(self, type='C'):

        if type == 'O' or type == 'H' or type =='L' or type == 'C':
            curr_price = self.getPrice(self.current_tick, type=type)
        elif type == 'T':
            curr_price = self.takeProfit
        elif type == 'S':
            curr_price = self.stopLoss

        if self.position == LONG:
            profit = (((curr_price - self.entryPrice) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
        elif self.position == SHORT:
            profit = (((self.entryPrice - curr_price) / self.OnePip) - self.spread) * self.calculateQuanity() * self.PipVal
        else:
            profit = 0.

        return profit


    def calculateQuanity(self):
        return self.MaxLossPerTrade / (self.PipVal * float(self.variables['PipsDivision']))

    def reset(self):
        self.current_tick = 0
        print("Starting Episode : ")

        # Positions
        self.buys = 0
        self.sells = 0
        self.lostBuys = 0
        self.lostSells = 0


        # Clear the variables
        self.balance = float(self.variables['StartingBalance'])
        self.starting_balance = self.balance
        self.portfolio = float(self.balance)
        self.profit = 0
        self.max_portfolio = -10000

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.history = []

        self.updateState()
        #print("State : ",self.state)
        return self.state
