# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:42:58 2021

@author: swis
"""
import torch
import torch.nn as nn
from mmd_pytorch import MMD_loss

def load_pretrained_net(model,path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_dict = torch.load(path,map_location = device)
    model_dict = model.state_dict()

    keys = []
    for k, v in pretrained_dict.items():
        keys.append(k)
    i = 0
    for k, v in model_dict.items():
        if v.size() == pretrained_dict[keys[i]].size():
            model_dict[k] = pretrained_dict[keys[i]]
            i = i + 1
    model.load_state_dict(model_dict)

class Alexnet_finetune(nn.Module):
    def __init__(self, num_class=5):
        super(Alexnet_finetune, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(#bottleneck is different from original architecture(which is 4096-4096-1000)
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
        )
        self.final_classifier = nn.Sequential( 
            nn.Linear(256, num_class)
        )

    def forward(self, input):
        input = self.features(input)
        input = input.view(input.size(0), -1)
        input = self.classifier(input)
        input = self.bottleneck(input)
        result = self.final_classifier(input)
        return result


class DA_Alex_FC1(nn.Module):
    def __init__(self, num_class=5):
        super(DA_Alex_FC1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
        )
        self.mmd = MMD_loss(kernel_type='rbf')

        self.final_classifier = nn.Sequential( 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_class)
        )
        
    def forward(self, source, target):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        mmdloss = 0.
        if self.training:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            mmdloss += self.mmd(source, target)          
        result = self.final_classifier(source)

        return result, mmdloss
    

class DA_Alex_FC2(nn.Module):
    def __init__(self, num_class=5):
        super(DA_Alex_FC2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        self.mmd = MMD_loss(kernel_type='rbf')
        self.final_classifier = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_class)
        )
        
    def forward(self, source, target):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        mmdloss = 0.
        if self.training:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            mmdloss += self.mmd(source, target)
        result = self.final_classifier(source)

        return result, mmdloss

class DA_Alex_FC3(nn.Module):
    def __init__(self, num_class=5):
        super(DA_Alex_FC3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True)
        )

        self.mmd = MMD_loss(kernel_type='rbf')
        self.final_classifier = nn.Sequential(
            nn.Linear(256, num_class)
        )
        
    def forward(self, source, target):
        source = self.features(source)
        source = source.view(source.size(0), -1)
        source = self.classifier(source)
        mmdloss = 0.
        if self.training:
            target = self.features(target)
            target = target.view(target.size(0), -1)
            target = self.classifier(target)
            mmdloss += self.mmd(source, target)
        result = self.final_classifier(source)

        return result, mmdloss



class LeNet_finetune(nn.Module):
    def __init__(self, num_class=5):
        super(LeNet_finetune, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)            
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*53*53, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120,84),
            nn.ReLU(inplace=True)     
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(84,num_class)
        )

    def forward(self, source):
        x = self.features(source)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        result = self.final_classifier(x)
        return result

class DA_LeNet_FC1(nn.Module):
    def __init__(self, num_class=5):
        super(DA_LeNet_FC1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*53*53, 120),
            nn.ReLU(inplace=True),   
        )
        self.mmd = MMD_loss(kernel_type='rbf')
        self.final_classifier = nn.Sequential( 
            nn.Linear(120,84),
            nn.ReLU(inplace=True),
            nn.Linear(84,num_class)
        )

    def forward(self, source, target):
        x = self.features(source)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        mmdloss = 0.
        if self.training:
            y = self.features(target)
            y = y.view(y.size(0), -1)
            y = self.classifier(y)
            mmdloss += self.mmd(x, y)
        result = self.final_classifier(x)
        return result, mmdloss

class DA_LeNet_FC2(nn.Module):
    def __init__(self, num_class=5):
        super(DA_LeNet_FC2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*53*53, 120),
            nn.ReLU(inplace=True),   
            nn.Linear(120,84),
            nn.ReLU(inplace=True)
        )
        self.mmd = MMD_loss(kernel_type='rbf')
        self.final_classifier = nn.Sequential( 
            nn.Linear(84,num_class)
        )

    def forward(self, source, target):
        x = self.features(source)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        mmdloss = 0.
        if self.training:
            y = self.features(target)
            y = y.view(y.size(0), -1)
            y = self.classifier(y)
            mmdloss += self.mmd(x, y)
        result = self.final_classifier(x)
        return result, mmdloss

cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_finetune(nn.Module):
    def __init__(self, feature, num_class=5):
        super().__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        output = self.feature(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output
    


def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)] # stride默认为1,即保持图像尺寸不变
        if batch_norm == True:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


def VGG11bn_finetune(num_class=5):
    return VGG_finetune(make_layers(cfg['A'], batch_norm=True),num_class=num_class)
def VGG13bn_finetune(num_class=5):
    return VGG_finetune(make_layers(cfg['B'], batch_norm=True),num_class=num_class)
def VGG16bn_finetune(num_class=5):
    return VGG_finetune(make_layers(cfg['C'], batch_norm=True),num_class=num_class)
def VGG19bn_finetune(num_class=5):
    return VGG_finetune(make_layers(cfg['D'], batch_norm=True),num_class=num_class)


class DA_VGG11bn_FC2(nn.Module):
    def __init__(self,num_class=5,batch_norm = True):
        super(DA_VGG11bn_FC2, self).__init__()
        cfg = {
    'A': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
        layers = []
        input_channel = 3
        for l in cfg['A']:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue
            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)] # stride默认为1,即保持图像尺寸不变
            if batch_norm == True:
                layers += [nn.BatchNorm2d(l)]
            layers += [nn.ReLU(inplace=True)]
            input_channel = l


        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            )
        self.mmd = MMD_loss(kernel_type='rbf')
        self.final_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4096, num_class)
            )
    
    def forward(self, source, target):
        x = self.features(source)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        mmdloss = 0.
        if self.training:
            y = self.features(target)
            y = y.view(y.size(0), -1)
            y = self.classifier(y)
            mmdloss += self.mmd(x, y)
        result = self.final_classifier(x)
        return result, mmdloss
