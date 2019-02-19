import pandas as pd
import numpy as np
import pickle

import ForexUtils as FXUtils

class ProcessedData:
    def __init__(self, inputSymbol, train_test_predict ="train"):
        sqlEngine = FXUtils.getSQLEngine()
        mlVariables = FXUtils.getMLVariables()
        self.symbol = inputSymbol
        self.train_test_split_ratio = float(mlVariables['RL-TrainTestSplit'])

        if train_test_predict == "train" or train_test_predict == "test":
            tableSize = FXUtils.count_records(mlVariables['MLDataTableName'],"Symbol = '" + inputSymbol + "'")
            splitPoint = int(tableSize * self.train_test_split_ratio)
            recordsAfterSplit = tableSize - splitPoint

        if train_test_predict == "train":
            tableName = mlVariables['MLDataTableName']
            sqlQuery = "SELECT * FROM " + tableName + " WHERE Symbol = '" + inputSymbol + "' LIMIT " + splitPoint
            symbolPickle = {}
            symbolPickle.scalers = {}

        elif train_test_predict == "test":
            tableName = mlVariables['MLDataTableName']
            sqlQuery = "SELECT * FROM " + tableName + " WHERE Symbol = '" + inputSymbol + "' LIMIT " + splitPoint + "," + recordsAfterSplit


        else:
            tableName = mlVariables['MLPredTableName']
            sqlQuery = "SELECT * FROM " + tableName + " WHERE Symbol = '"+inputSymbol+"' AND Status = 'DATA_READY'"

        self.pickle = symbolPickle
        rawData = pd.read_sql_query(sqlQuery,sqlEngine)
        rawData.set_index('Time')
        rawData['CloseShifted'] = rawData['Close'].shift(1)
        rawData.dropna(axis=0)

        df = rawData[['CloseShifted']].copy()
        df['Bar_HC'] = rawData['High'] - rawData['Close']
        df['Bar_HO'] = rawData['High'] - rawData['Open']
        df['Bar_HL'] = rawData['High'] - rawData['Low']
        df['Bar_CL'] = rawData['Close'] - rawData['Low']
        df['Bar_CO'] = rawData['Close'] - rawData['Open']
        df['Bar_OL'] = rawData['Open'] - rawData['Low']
        df['Volume'] = rawData['Volume']

        self.df = df
        self.rawData = rawData

    def addSimpleFeatures(self):
        mlVariables = FXUtils.getMLVariables()
        addOnFeaturesList = mlVariables['Model5AddonFeatures'].split(',')
        addOnNumericalExcepList = mlVariables['Model5AddonFeaturesNumerical'].split(',')
        addOnNumClassesList = mlVariables['Model5AddonFeaturesNumClasses'].split(',')
        print("features : ", addOnFeaturesList)
        print("numClasses : ",addOnNumClassesList)
        for i,col in enumerate(addOnFeaturesList):
            if col not in addOnNumericalExcepList:
                self.df[col] = self.rawData[col]
                self.df, colArr = FXUtils.getCategoricalColumnsFromDF(self.df, col, int(addOnNumClassesList[i]))
            else:
                self.df[col] = self.rawData[col]
                self.df[col] = self.df[col].astype('float64')

    def apply_normalization(self):
        for col in self.df.columns.values.tolist():
            self.pickle['scalers'][col] = FXUtils.getNormalizedData(self.df[[col]])
            self.df[col] = self.pickle['scalers'][col].transform(self.df[col])
        pickle.dump(self.pickle,open("Pickles\\" + self.symbol + "-Pickle.pkl","wb"))




# testSymbol = "EURUSD"
# forexData = ProcessedData(testSymbol)
# forexData.addSimpleFeatures()
# print("Dataframe : ",forexData.df)

##############