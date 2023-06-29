import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Dense
from keras.activations import relu,sigmoid
from keras.losses import BinaryCrossentropy

train=pd.read_csv('train.csv')
pd.set_option('display.max_columns',None)
#print(train.shape)
print(train.keys())
#print(train.head(100))

#take group name from passengerId
#ignore home planet
#cryosleep if nan then false
#ignore cabin code
#ignore destination
#consider age,if nan then average
#consider vip, if nan then false
#amounts spend, if nan then 0
#ignore name

np_array=train.values
#print(np_array.shape)

#print(np_array[:4])
#print(np_array[:,0])
np_array[:,0]=[int(i[:4]) for i in np_array[:,0]]
print(np_array[:,0])

np_array[:,2]=[1 if i==True else 0 for i in np_array[:,2]]

np_array[:,6]=[1 if i==True else 0 for i in np_array[:,6]]

y_train=np_array[:,-1]
y_train=[1 if i==True else 0 for i in y_train]
y_train=np.reshape(y_train,(-1,1))

x_train=np.delete(np_array,[1,3,4,12,13],axis=1)
x_train=x_train.astype(float)

print(x_train[0])
for i in range(4,9):
    x_train[:,i]=[i if ~np.isnan(i) else 0 for i in x_train[:,i]]


x_train[:,2]=[i if ~np.isnan(i) else 20 for i in x_train[:,2]]

avg=np.average(x_train,axis=0)

std=np.std(x_train,axis=0)
x_train=(x_train-avg)/std
print(x_train.shape)
print(x_train[:5])

print(y_train)

model=Sequential([
    Dense(units=5,activation=relu,kernel_regularizer=l2(.1)),
    Dense(units=1,activation=sigmoid,kernel_regularizer=l2(.1))
])

model.compile(
    loss=BinaryCrossentropy(),
    optimizer=Adam(.001)
)

model.fit(x_train,y_train,epochs=30)

test_Data=pd.read_csv('test.csv')
print(test_Data.keys())

total_set=test_Data.values
passengerId=total_set[:,0].reshape((-1,1))


x_test=np.delete(total_set,[1,3,4,12],axis=1)
print(x_test.shape)
x_test[:,0]=[int(i[:4]) for i in x_test[:,0]]
x_test[:,1]=[1 if i==True else 0 for i in x_test[:,1]]
x_test[:,3]=[1 if i==True else 0 for i in x_test[:,3]]

x_test=x_test.astype(float)
x_test[:,2]=[i if ~np.isnan(i) else 20 for i in x_test[:,2]]
for i in range(4,9):
    x_test[:,i]=[i if ~np.isnan(i) else 0 for i in x_test[:,i]]

x_test=(x_test-avg)/std
print(x_test)
y_test=model.predict(x_test)

y_test=[True if i>=0.5 else False for i in y_test]
print(y_test)

y_test=np.array(y_test).reshape((-1,1))
output=np.concatenate([passengerId,y_test],axis=1)

file="output.csv"
header="PassengerId,Transported"
np.savetxt(file,output,delimiter=",",header=header,comments="",fmt='%s')





