import numpy as np
import pandas as pd
import ForexUtils as FXU

print(np.array([1,2,3,4,5]))

print("Commit 2")

print("Commit 3 and pushed")

#for i in range(10):
#    print("i : ",i)

ForexDBParameters = dict(line.strip().split('=') for line in open('forexDB.properties') if not line.startswith('#') and not line.startswith('\n'))
print("UserName : ",ForexDBParameters['username'],"password : ",ForexDBParameters['password'])
#print(ForexDBParameters)


df = pd.DataFrame(data={'column' : [1,2,3,4,6,8]})

df['shifted'] = df['column'] - df['column'].shift(1)
df = df.dropna(axis=0)
#print(df)

MLDataTableCount = FXU.count_records("mlmqldata10", "Symbol = 'EURUSD'")
print("Data Table Count : ",MLDataTableCount)

df = pd.DataFrame(data={'col1': [1,2,3,4,5,6], 'col2': [np.nan,5,6,7,8,9]})
df = df.dropna(axis=0)
print(df)

#FXU.drop_table('droptabletest')

#actions_table_details = {'name':'metaactions', 'col':['Action', 'Time'], 'type':['VARCHAR(20)', 'DATETIME'], 'null': [False,False]}
#FXU.create_table_db(actions_table_details)
df = pd.DataFrame({'BoolCol': [1,2,3,4,5]}, index=[10,20,30,40,50])

print("index : ",df.index[1])
