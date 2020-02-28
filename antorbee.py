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


# if __name__ == '__main__':
    data_dir = 'data_ant'
    image_size = 224

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

'''


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





'''



