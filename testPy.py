import numpy as np

print(np.array([1,2,3,4,5]))

print("Commit 2")

print("Commit 3 and pushed")

for i in range(10):
    print("i : ",i)

ForexDBParameters = dict(line.strip().split('=') for line in open('forexDB.properties') if not line.startswith('#') and not line.startswith('\n'))
print("UserName : ",ForexDBParameters['username'],"password : ",ForexDBParameters['password'])
#print(ForexDBParameters)
