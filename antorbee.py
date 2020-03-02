import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os


data_dir = 'data_ant'
image_size = 224
num_classes = 0 # 初始化

def imshow(inp,title = None):
# inp为一个张量 channels*image_width*image_height
# as always,the image is image_width*image_height*channels

    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    inp = std*inp +mean
    inp = np.clip(inp,0,1)
#make sure that inp is in 0,1
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)




# ok lets build our first Cov
depth = [4,8]
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3,4,5,padding=2)
        # 定义一个卷积层，输入通道为3，输出通道为4，窗口大小为5，padding为2
        self.conv2 = nn.Conv2d(depth[0],depth[1],5,padding=2)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(image_size//4*image_size//4*depth[1],512)
        self.fc2 = nn.Linear(512,num_classes)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1,image_size//4*image_size//4*depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x

    def retrieve_features(self,x):
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1,feature_map2)



def rightness(predictions,labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，
    batch_size行10列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions,dim=1).indices
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights,len(labels)








if __name__ == '__main__':


    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                         transforms.Compose([transforms.RandomResizedCrop(image_size),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                                  [0.229, 0.224, 0.225])]))

    # 将多个transform组合起来使用。
    # RandomCrop(size, padding=0)
    # 切割中心点的位置随机选取
    # 随机水平翻转给定的PIL.Image,概率为0.5
    # 先将给定的PIL.Image随机切，然后再resize成给定的size大小。

    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
                                       )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

    num_classes = len(train_dataset.classes)

    use_cuda = torch.cuda.is_available()

    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    # next(iterobject,defalt)函数的第一个参数是一个可迭代对象,第二个参数可以不写。
    # 不写的时候，如果可迭代对象的元素取出完毕，会返回StopIteration。如果第二个参数写一个其他元素，则可迭代对象迭代完毕后，会一直返回写的那个元素。

    images, labels = next(iter(train_loader))
    out = torchvision.utils.make_grid(images)
    imshow(out, title=[train_dataset.classes[x] for x in labels])
    # make_grid的作用是将若干幅图像拼成一幅图像。其中padding的作用就是子图像与子图像之间的pad有多宽


    a = input("use self model(1) or resnet?(2) resnetfix(3) ")
    if a == '1':
        net =ConvNet()
        net = net.cuda() if use_cuda else net
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),lr =0.001,momentum=0.9)
        record = []
        num_epochs = 10
        net.train(True)

        best_model = net
        best_r = 0.0
        for epoch in range(num_epochs):
            train_rights = []
            train_losses = []
            for batch_idx,(data,target) in enumerate(train_loader):
                data,target = Variable(data),Variable(target)
                if use_cuda:
                    data,target = data.cuda(),target.cuda()
                output = net(data) # predict for once

                loss = criterion(output,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                right = rightness(output,target)
                train_rights.append(right)

                loss = loss.cpu() if use_cuda else loss
                train_losses.append(loss.data.numpy())

            train_r = ((sum([tup[0] for tup in train_rights])),(sum([tup[1] for tup in train_rights])))
            net.eval()
            test_loss = 0
            correct = 0
            vals = []
            # 对测试数据集进行循环
            for data, target in val_loader:
                data, target = Variable(data),Variable(target)
                # 如果GPU可用，就把数据加载到GPU中
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = net(data)  # 将特征数据喂入网络，得到分类的输出
                val = rightness(output, target)  # 获得正确样本数以及总样本数
                vals.append(val)  # 记录结果

            # 计算准确率
            val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
            val_ratio = 1.0 * val_r[0] / val_r[1]


            if val_ratio > best_r:
                best_r = val_ratio
                best_model = copy.deepcopy(net)
            print('训练周期: {} \tLoss: {:.6f}\t训练正确率: {:.2f}%, 校验正确率: {:.2f}%'.format(
                epoch, np.mean(train_losses), 100. * train_r[0].numpy() / train_r[1], 100. * val_r[0].numpy() / val_r[1]))
            record.append([np.mean(train_losses), train_r[0].numpy() / train_r[1], val_r[0].numpy() / val_r[1]])

        x = [x[0] for x in record]
        y = [1 - x[1] for x in record]
        z = [1 - x[2] for x in record]
        # plt.plot(x)
        plt.figure(figsize=(10, 7))
        plt.plot(y)
        plt.plot(z)
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')


        plt.show()

    elif a=='2':



        net = models.resnet18(pretrained=True)

        net = net.cuda() if use_cuda else net
        #加上下面这个就是固定值训练
        #for param in net.parameters():
        #    param.requires_grad = False
        num_ftrs = net.fc.in_features
        # 读取最后线性层的输入单元数，这是前面各层卷积提取到的特征数量
        net.fc = nn.Linear(num_ftrs,2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),lr = 0.0001,momentum=0.9)

        # data,target = Variable(data),Variable(target)
        # if use_cuda:
        #     data,target = data.cuda(),target.cuda()

        record = []
        num_epochs=20
        net.train(True)
        best_model = net
        best_r = 0.0
        for epoch in range(num_epochs):
            train_rights = []
            train_losses = []
            for batch_idx,(data,target) in enumerate(train_loader):
                data,target = data.clone().detach().requires_grad_(False),target.clone().detach()
                if use_cuda:
                    data,target = data.cuda(),target.cuda()
                output = net(data)
                loss = criterion(target,output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                right = rightness(output,target)
                train_rights.append(right)
                loss = loss.cpu() if use_cuda else loss
                #if use cuda then you have to make them come back
                train_losses.append(loss.data.numpy())
            train_r = (sum([tup[0] for tup in train_rights]),sum([tup[1] for tup in train_rights]))
            net.eval()  # 标志模型当前为运行阶段
            test_loss = 0
            correct = 0
            vals = []
            # 对测试数据集进行循环
            for data, target in val_loader:
                # 如果存在GPU则将变量加载到GPU中
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
                output = net(data)  # 将特征数据喂入网络，得到分类的输出
                val = rightness(output, target)  # 获得正确样本数以及总样本数
                vals.append(val)  # 记录结果

            # 计算准确率
            val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
            val_ratio = 1.0 * val_r[0].numpy() / val_r[1]

            if val_ratio > best_r:
                best_r = val_ratio
                best_model = copy.deepcopy(net)
            # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
            print('训练周期: {} \tLoss: {:.6f}\t训练正确率: {:.2f}%, 校验正确率: {:.2f}%'.format(
                epoch, np.mean(train_losses), 100. * train_r[0].numpy() / train_r[1],
                                              100. * val_r[0].numpy() / val_r[1]))
            record.append([np.mean(train_losses), train_r[0].numpy() / train_r[1], val_r[0].numpy() / val_r[1]])
        x = [x[0] for x in record]
        y = [1 - x[1] for x in record]
        z = [1 - x[2] for x in record]
        # plt.plot(x)
        plt.figure(figsize=(10, 7))
        plt.plot(y)
        plt.plot(z)
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        # plt.show()

    elif a=='3':
        net = models.resnet18(pretrained=True)

        net = net.cuda() if use_cuda else net
        # 加上下面这个就是固定值训练
        for param in net.parameters():
           param.requires_grad = False
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
        net.fc = net.fc.cuda() if use_cuda else net.fc

        criterion = nn.CrossEntropyLoss()  # Loss函数的定义
        # 仅将线性层的参数放入优化器中
        optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)

        record = []  # 记录准确率等数值的容器

        # 开始训练循环
        num_epochs = 20
        net.train(True)  # 给网络模型做标记，标志说模型在训练集上训练
        best_model = net
        best_r = 0.0
        for epoch in range(num_epochs):
            # optimizer = exp_lr_scheduler(optimizer, epoch)
            train_rights = []  # 记录训练数据集准确率的容器
            train_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
                data, target = data.clone().detach().requires_grad_(True), target.clone().detach()  # data为图像，target为标签
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = net(data)  # 完成一次预测
                loss = criterion(output, target)  # 计算误差
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播
                optimizer.step()  # 一步随机梯度下降
                right = rightness(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                train_rights.append(right)  # 将计算结果装到列表容器中
                loss = loss.cpu() if use_cuda else loss
                train_losses.append(loss.data.numpy())

            # train_r为一个二元组，分别记录训练集中分类正确的数量和该集合中总的样本数
            train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

            # 在测试集上分批运行，并计算总的正确率
            net.eval()  # 标志模型当前为运行阶段
            test_loss = 0
            correct = 0
            vals = []
            # 对测试数据集进行循环
            for data, target in val_loader:
                data, target = data.clone().detach().requires_grad_(False), target.clone().detach()
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                output = net(data)  # 将特征数据喂入网络，得到分类的输出
                val = rightness(output, target)  # 获得正确样本数以及总样本数
                vals.append(val)  # 记录结果

            # 计算准确率
            val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
            val_ratio = 1.0 * val_r[0].numpy() / val_r[1]

            if val_ratio > best_r:
                best_r = val_ratio
                best_model = copy.deepcopy(net)
            # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
            print('训练周期: {} \tLoss: {:.6f}\t训练正确率: {:.2f}%, 校验正确率: {:.2f}%'.format(
                epoch, np.mean(train_losses), 100. * train_r[0].numpy() / train_r[1],
                                              100. * val_r[0].numpy() / val_r[1]))
            record.append([np.mean(train_losses), train_r[0].numpy() / train_r[1], val_r[0].numpy() / val_r[1]])
        x = [x[0] for x in record]
        y = [1 - x[1] for x in record]
        z = [1 - x[2] for x in record]
        # plt.plot(x)
        plt.figure(figsize=(10, 7))
        plt.plot(y)
        plt.plot(z)
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.show()


    times = 10
    results = {}
    models_name = ['conv','transfer_pre','transfer_fixed']
    for model in models_name:
        if model=='conv':
            net = ConvNet()
            net = net.cuda() if use_cuda else net
            criterion = nn.CrossEntropyLoss()  # Loss函数的定义
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        if model == 'transfer_pretrain':
            net = models.resnet18(pretrained=True)
            net = net.cuda() if use_cuda else net
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 2)
            net.fc = net.fc.cuda() if use_cuda else net.fc

            criterion = nn.CrossEntropyLoss()  # Loss函数的定义
            optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        if model == 'transfer_fixed':
            net = models.resnet18(pretrained=True)
            net = net.cuda() if use_cuda else net
            for param in net.parameters():
                param.requires_grad = False
            num_ftrs = net.fc.in_features
            net.fc = nn.Linear(num_ftrs, 2)
            net.fc = net.fc.cuda() if use_cuda else net.fc

            criterion = nn.CrossEntropyLoss()  # Loss函数的定义
            optimizer = optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)
        for time in range(times):
            print(model, time)
            record = []  # 记录准确率等数值的容器
            # 开始训练循环
            num_epochs = 20
            net.train(True)  # 给网络模型做标记，标志说模型在训练集上训练
            for epoch in range(num_epochs):
                # optimizer = exp_lr_scheduler(optimizer, epoch)本文件是集智AI学园http://campus.swarma.org 出品的“火炬上的深度学习”第IV课的配套源代码
                train_rights = []  # 记录训练数据集准确率的容器
                train_losses = []
                for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
                    data, target = data.clone().detach().requires_grad_(
                        True), target.clone().detach()  # 将data为图像，target为标签
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()
                    output = net(data)  # 完成一次预测
                    loss = criterion(output, target)  # 计算误差
                    optimizer.zero_grad()  # 清空梯度
                    loss.backward()  # 反向传播
                    # loss = loss.cpu() if use_cuda else loss
                    optimizer.step()  # 一步随机梯度下降
                    right = rightness(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                    train_rights.append(right)  # 将计算结果装到列表容器中
                # train_r为一个二元组，分别记录训练集中分类正确的数量和该集合中总的样本数
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

                # 在测试集上分批运行，并计算总的正确率
                net.eval()  # 标志模型当前为运行阶段
                test_loss = 0
                correct = 0
                vals = []
                # 对测试数据集进行循环
                for data, target in val_loader:
                    data, target = data.clone().detach().requires_grad_(False), target.clone().detach()
                    if use_cuda:
                        data, target = data.cuda(), target.cuda()
                    output = net(data)  # 将特征数据喂入网络，得到分类的输出
                    val = rightness(output, target)  # 获得正确样本数以及总样本数
                    vals.append(val)  # 记录结果

                # 计算准确率
                val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))

                # val_ratio = 1.0 * val_r[0].numpy() / val_r[1]

                # if val_ratio > best_r:
                #     best_r = val_ratio
                #     best_model = copy.deepcopy(net)
                # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
                # print('训练周期: {:.6f}\t训练正确率: {:.2f}%, 校验正确率: {:.2f}%'.format(
                #     epoch, 100. * train_r[0].numpy() / train_r[1], 100. * val_r[0].numpy() / val_r[1]))
                # record.append([train_r[0].numpy() / train_r[1], val_r[0].numpy() / val_r[1]])
            # 将结果加入results中，record记载了每一个打印周期的训练准确度和校验准确度
            results[(model, time)] = record

    for model in models_name:
        errors = []
        for time in range(times):
            rs = results[(model,time)]
            errors.append(rs)
            #rs is ratio_train/ratio_val
        aa = np.array(errors)
        avg = np.mean(aa, 0)
        plt.figure(figsize=(10, 7))
        plt.title(model)
        plt.plot([1 - i for i in avg])
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.show()



