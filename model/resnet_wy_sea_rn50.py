import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('norm1', nn.BatchNorm2d(outchannel)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(outchannel))
        ]))
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv3', nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)),
                ('norm3', nn.BatchNorm2d(outchannel))
        ]))

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    # def __init__(self, ResidualBlock, num_classes=10):
    def __init__(self, ResidualBlock, num_classes):
        super(ResNet, self).__init__()
        self.inchannel = 64
        
        # below is WY's modification that follows the standard torchvision resnet
        # the kernal size is corrected to 7, 
        # the strade is corrected to be 2, 
        # the padding is corrected to be 3,
        # and the lacking maxpooling layer is added
        self.features = nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0',nn.BatchNorm2d(64)),
            ('relu0',nn.ReLU(inplace=True)),
            ('pool1',nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # WY: note that out channel number to be [64, 128, 256, 512] for standard resnet
        self.features.add_module('ResidualBlock1',self.make_layer(ResidualBlock, 64, 3, stride=1))
        self.features.add_module('ResidualBlock2',self.make_layer(ResidualBlock, 128, 4, stride=2))
        self.features.add_module('ResidualBlock3',self.make_layer(ResidualBlock, 256, 6, stride=2))
        self.features.add_module('ResidualBlock4',self.make_layer(ResidualBlock, 512, 3, stride=2))

        # below is copied from https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # WY: note that input channel should be 512 for standard resnet18 (2048 for resnet50)
        # WY: note that the input channel could be 64*64=4096=512*8 for seagate's dataset
        self.classifier = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out=F.log_softmax(out,dim=1)
        return out

# below are WY's modification
# remember to key in the args num_classes = 10 for seagate's datasets
def ResNet50(num_classes=10):
    return ResNet(ResidualBlock, num_classes=num_classes)

# # below are original code lines
# def ResNet18():
#     return ResNet(ResidualBlock)




def Quantized_resnet(pre_model, args):

    pre_model.features.conv0.weight.requires_grad=False

    weights=[p for n, p in pre_model.named_parameters() if 'classifier.weight' in n]
    biases=[pre_model.classifier.bias]

    #layers that need to be quantized
    weights_to_be_quantized = [p for n, p in pre_model.named_parameters() if 'conv' in n and ('ResidualBlock' in n)]

    # weights and biases of batch normlization layer
    bn_weights = [p for n, p in pre_model.named_parameters() if 'norm' in n and 'weight' in n]
    bn_biases = [p for n, p in pre_model.named_parameters() if 'norm' in n and 'bias' in n]

    params=[
        {'params': weights,'weight_decay': 5.0e-4},
        {'params': weights_to_be_quantized},
        {'params': biases},
        {'params': bn_weights},
        {'params': bn_biases}
    ]

    optimizer=optim.Adam(params, lr=args.lr)
    loss_fun=nn.CrossEntropyLoss()

    return pre_model,loss_fun,optimizer

