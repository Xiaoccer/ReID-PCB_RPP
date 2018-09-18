import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.constant(m.weight.data, 1)
        init.constant(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []

        add_block += [nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False)]
        add_block += [nn.BatchNorm2d(num_bottleneck)]
        if relu:
            #add_block += [nn.LeakyReLU(0.1)]
            add_block += [nn.ReLU(inplace=True)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# Define the RPP layers
class RPP(nn.Module):
    def __init__(self):
        super(RPP, self).__init__()
        self.part = 6
        add_block = []
        add_block += [nn.Conv2d(2048, 6, kernel_size=1, bias=False)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        norm_block = []
        norm_block += [nn.BatchNorm2d(2048)]
        norm_block += [nn.ReLU(inplace=True)]
        # norm_block += [nn.LeakyReLU(0.1, inplace=True)]
        norm_block = nn.Sequential(*norm_block)
        norm_block.apply(weights_init_kaiming)

        self.add_block = add_block
        self.norm_block = norm_block
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        w = self.add_block(x)
        p = self.softmax(w)
        y = []
        for i in range(self.part):
            p_i = p[:, i, :, :]
            p_i = torch.unsqueeze(p_i, 1)
            y_i = torch.mul(x, p_i)
            y_i = self.norm_block(y_i)
            y_i = self.avgpool(y_i)
            y.append(y_i)

        f = torch.cat(y, 2)
        return f


# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num):
        super(PCB, self).__init__()

        self.part = 6
        # resnet50
        resnet = models.resnet50(pretrained=True)
        # remove the final downsample
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        modules = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        # define 6 classifiers
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(ClassBlock(2048, class_num, True, 256))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x[:, :, i, :]
            part[i] = torch.unsqueeze(part[i], 3)
            # print part[i].shape
            predict[i] = self.classifiers[i](part[i])

        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

    def convert_to_rpp(self):
        self.avgpool = RPP()
        return self


class PCB_test(nn.Module):
    def __init__(self, model, featrue_H=False):
        super(PCB_test, self).__init__()
        self.part = 6
        self.featrue_H = featrue_H
        self.backbone = model.backbone
        self.avgpool = model.avgpool
        self.classifiers = nn.ModuleList()
        for i in range(self.part):
            self.classifiers.append(model.classifiers[i].add_block)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)

        if self.featrue_H:
            part = {}
            predict = {}
            # get six part feature batchsize*2048*6
            for i in range(self.part):
                part[i] = x[:, :, i, :]
                part[i] = torch.unsqueeze(part[i], 3)
                predict[i] = self.classifiers[i](part[i])

            y = []
            for i in range(self.part):
                y.append(predict[i])
            x = torch.cat(y, 2)
        f = x.view(x.size(0), x.size(1), x.size(2))
        return f


# debug model structure

net = PCB(751)
net = net.convert_to_rpp()
print(net)
input = Variable(torch.FloatTensor(8, 3, 7, 7))
output = net(input)
# print(output[0].shape)
