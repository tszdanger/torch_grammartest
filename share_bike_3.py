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
    rides = pd.concat([rides,dummies],axis=1)

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
    #这里注意到我们把cnt也变了，以后可能要取整

test_data = data[-21*24:]
train_data = data[:-21*24]
target_fields = ['cnt','casual','registered']
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


# 构建神经网络
# features is (16875,56)

input_size = features.shape[1]
hidden_size = 10
output_size = 1
batch_size = 128
neu = torch.nn.Sequential(
    torch.nn.Linear(input_size,hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size,output_size),
)
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(neu.parameters(),lr=0.01)


for i in range(300):
    batch_loss = []
    for start in range(0,len(X),batch_size):
        if (start+ batch_size)<len(X):
            end = start + batch_size
        else:
            end = len(X)
        xx = Variable(torch.FloatTensor(X[start:end]))
        yy = Variable(torch.FloatTensor(Y[start:end]))
        predict = neu(xx)
        loss = cost(predict,yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    if (i%100 == 0):
        losses.append(np.mean(batch_loss))
        print(i,np.mean(batch_loss))

# plt.plot(np.arange(len(losses))*100,losses)
# plt.xlabel('epoch')
# plt.ylabel('MSE')
# plt.show()



# then lets test it
targets = test_targets['cnt']
targets = targets.values.reshape([len(targets),1])
targets = targets.astype(float)

x = Variable(torch.FloatTensor(test_features.values))
y = Variable(torch.FloatTensor(targets))

predict = neu(x)
predict = predict.data.numpy()
fig,ax = plt.subplots(figsize = (10,7))
mean,std = scaled_features['cnt']

# ax.plot(predict*std+mean,label = 'prediction')
# ax.plot(targets*std+mean,label = 'Data')
#
# ax.legend()
# ax.set_xlabel('time')
# ax.set_ylabel('counts')
# dates = pd.to_datetime(rides.loc[test_data.index]['dteday'])
# dates = dates.apply(lambda d: d.strftime('%b %d'))
# ax.set_xticks(np.arange(len(dates))[12::24])
# _=ax.set_xticklabels(dates[12::24],rotation=45)
#
#
# plt.show()

def feature(X,net):
    X = Variable(torch.from_numpy(X).type(torch.FloatTensor),requires_grad = False)
    dic = dict(net.named_parameters())
    weights = dic['0.weight']
    biases = dic['0.bias']
    h = torch.sigmoid(X.mm(weights.t())+biases.expand([len(X),len(biases)]))
    return h

bool1 = rides['dteday'] == '2012-12-22'
bool2 = rides['dteday'] == '2012-12-23'
bool3 = rides['dteday'] == '2012-12-24'


bools = [any(tup) for tup in zip(bool1,bool2,bool3)]
# any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False/0/''，则返回 False，如果有一个为 True，则返回 True
#zip 返回的也是对象了

subset = test_features.loc[rides[bools].index]
subtargets = test_targets.loc[rides[bools].index]
subtargets = subtargets['cnt']
subtargets = subtargets.values.reshape((len(subtargets),1))

results = feature(subset.values,neu).data.numpy()
predict = neu(Variable(torch.FloatTensor(subset.values))).data.numpy()
mean,std = scaled_features['cnt']
predict = predict*std+mean
subtargets = subtargets*std+mean

# fig, ax = plt.subplots(figsize = (8, 6))
# ax.plot(results[:,:],'.:',alpha = 0.3)
# ax.plot((predict - min(predict)) / (max(predict) - min(predict)),'bs-',label='Prediction')
# ax.plot((subtargets - min(predict)) / (max(predict) - min(predict)),'ro-',label='Real')
# ax.plot(results[:, 3],':*',alpha=1, label='Neuro 4')
#
# ax.set_xlim(right=len(predict))
# ax.legend()
# plt.ylabel('Normalized Values')
#
# dates = pd.to_datetime(rides.loc[subset.index]['dteday'])
# dates = dates.apply(lambda d: d.strftime('%b %d'))
# ax.set_xticks(np.arange(len(dates))[12::24])
# _ = ax.set_xticklabels(dates[12::24], rotation=45)

dic = dict(neu.named_parameters())
# weights = dic['2.weight']
# plt.plot(weights.data.numpy()[0],'o-')
# plt.xlabel('Input Neurons')
# plt.ylabel('Weight')

weights = dic['0.weight']
plt.plot(weights.data.numpy()[6,:],'o-')
plt.xlabel('Input Neurons')
plt.ylabel('Weight')

plt.show()