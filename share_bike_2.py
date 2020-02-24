import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim

rides = pd.read_csv('data_bike/hour.csv')
rides.head()
counts = rides['cnt'][:50]

dummy_fields = ['season','weathersit','mnth','hr','weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each],prefix=each)
    #这里挨个转换独热编码,转换完了一个就粘贴一个
    pd.concat([rides,dummies],axis=1)

fiels_to_drop = ['instant', 'dteday', 'season', 'weathersit','weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fiels_to_drop,axis=1)

#to deal with the data type

quant_features = ['cnt','temp','hum','windspeed']
scaled_features = {}
for each in quant_features:
    mean,std = data[each].mean(),data[each].std()
    scaled_features[each] = [mean,std]
    data.loc[:,each] = (data[each]-mean)/std
    # in this step we change the memory of data
    #这一步我们改变了data的值，以后想要还原都必须靠scaled_features


test_data = data[-21*24:]
train_data = data[:-21*24]
target_fields = ['cnt','casual','registerd']
#we think that these three ways are the result we want to predict
features,targets = train_data.drop(target_fields,axis=1),train_data[target_fields]
test_features,test_targets = test_data.drop(target_fields,axis=1),test_data[target_fields]


X = features.values
#change dataframe to numpy
Y = targets['cnt'].values
Y = Y.astype(float)
Y = np.reshape(Y,[len(Y),1])
#这一步reshape的道理是避免后续去转
#Y = np.reshape(Y,(len(Y),1))也是一样的

losses = []
