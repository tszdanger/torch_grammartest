import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim

# in this script we will develop a NN
# it's very easy and the only magic point is on the operation of the data


data_path = '201912-capitalbikeshare-tripdata.csv'
rides = pd.read_csv('201912-capitalbikeshare-tripdata/201912-capitalbikeshare-tripdata.csv')
rides.head()

counts = rides['Duration'][:50]
x = np.arange(len(counts))
#x is like [0 1 2 3 ...]
y = np.array(counts)

plt.figure(figsize=(10,8))

plt.plot(x,y,'o')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
#now let's use NN

x = Variable(torch.linspace(1,len(counts),len(counts)).type(torch.FloatTensor))
# or you can use
x = Variable(torch.FloatTensor(np.arange(len(counts),dtype = float)))
y = Variable(torch.FloatTensor(np.array(counts,dtype = float)))

n_size = 10

weight1 = Variable(torch.randn(1,n_size),requires_grad = True)
biase1 = Variable(torch.randn(n_size,1),requires_grad = True)
weight2 = Variable(torch.randn(n_size,1),requires_grad = True)

lr = 0.1
losses = []
for i in range(10000):
    # hidden = x.mm(weight1)+biase1
    hidden = x.expand(n_size,len(x)).t()*weight1.expand(len(x),n_size)+biase1.expand(len(x),n_size)


    hidden = torch.sigmoid(hidden)
    pred = hidden.mm(weight2)
    loss = torch.mean((pred-y)**2)
    losses.append(loss.data.numpy())
    if i%1000 == 0:
        print('loss is ',loss)

    loss.backward()
    weight1.data.add_(-lr*weight1.grad.data)
    weight2.data.add_(-lr*weight2.grad.data)
    biase1.data.add_(-lr*biase1.grad.data)

    weight1.grad.data.zero_()
    weight2.grad.data.zero_()
    biase1.grad.data.zero_()

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('loss')



x_data = x.data.numpy()


plt.figure(figsize=(10,8))

# plt.plot(x_train.data.numpy(),y_train.data.numpy(),'o')
xplot, = plt.plot(x_data,y.data.numpy(),'o')
#the origin data
yplot, = plt.plot(x_data,pred.data.numpy())

plt.legend([xplot,yplot],['DATA','prediction'])

plt.xlabel('x')
plt.ylabel('y')
plt.show()