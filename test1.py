import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
# a simple test for module apt

x = Variable(torch.linspace(1,100,100).type(torch.FloatTensor))

rand = Variable(torch.randn(100))*10

y = x+rand

x_train =x[:-10]
y_train = y[:-10]
x_test = x[-10:]
y_test = y[-10:]
#
# plt.figure(figsize=(10,8))
#
# plt.plot(x_train.data.numpy(),y_train.data.numpy(),'o')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


a = Variable(torch.rand(1),requires_grad = True)
#requires_grad makes sure that it gains the back_pro infomation during the whole process
b = Variable(torch.rand(1),requires_grad = True)

learning_rate = 0.0001

for i in range(10000):
    a1 = a.expand_as(x_train)
    b1 = b.expand_as(x_train)
    prediction = a1* x_train + b1

    loss = torch.mean((prediction - y_train)**2)

    if i%100 == 0:
        print('loss is {}'.format(loss))
        print('a is ',a)
        print('b is ',b)

    loss.backward()
    a.data.add_(-learning_rate*a.grad.data)
    b.data.add_(-learning_rate*b.grad.data)
    a.grad.data.zero_()
    b.grad.data.zero_()

x_data = x_train.numpy()


plt.figure(figsize=(10,8))

# plt.plot(x_train.data.numpy(),y_train.data.numpy(),'o')
xplot, = plt.plot(x_data,y_train.data.numpy(),'o')
yplot, = plt.plot(x_data,a.data.numpy()*x_data+b.data.numpy())

str1 = str(a.data.numpy()[0]) + 'x +'+str(b.data.numpy()[0])
plt.legend([xplot,yplot],['DATA',str1])

plt.xlabel('x')
plt.ylabel('y')
plt.show()



