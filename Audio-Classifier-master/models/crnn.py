import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models as models
import string
import numpy as np

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, nc, nh, n_rnn=2, nclass=10, leakyRelu=False):
        super(CRNN, self).__init__()

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]101 5 512

        # rnn features
        output = self.rnn(conv)

        return output


cfg = {'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']}
def _make_layers(cfg):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)

class CRNN_GRU(nn.Module):
    def __init__(self, num_classes=10, backend='VGG11', rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0, seq_len=1001):
        super(CRNN_GRU,self).__init__()
        # vgg11 feature extraction
        self.num_classes = num_classes
        self.seq_len=seq_len
        #self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)
        self.features = _make_layers(cfg["VGG11"])
        # rnn GRU classifer
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn = nn.GRU(input_size=512, hidden_size=self.rnn_hidden_size, num_layers=self.rnn_num_layers,
                          batch_first=False, dropout=rnn_dropout, bidirectional=True).cuda()
        self.classifier = nn.Sequential(nn.Dropout(),  # 防止过拟合
            #nn.Linear(rnn_hidden_size*2, 10)
            nn.Linear(3840, 10)
        )

    def forward(self, input):
        out = self.features(input).cuda()
        out = out.squeeze(2)
        hidden = self.init_hidden(out.size(0), next(self.parameters()).is_cuda)
        out = self.features_to_sequence(out).cuda()
        seq, hidden = self.rnn(out, hidden)
        seq = seq.view(seq.size(1), -1).cuda()
        seq = self.classifier(seq).cuda()
        seq = F.log_softmax(seq, dim=1).cuda()
        return seq

    def init_hidden(self, batch_size, gpu=True):
        h0 = Variable(torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size))
        if gpu:
            h0 = h0.cuda()
        return h0

    def features_to_sequence(self, features):
        #b, c, h, w = features.size()
        #assert h == 1, "the height of out must be 1"
        #features = features.squeeze(2)# 10 512 12
        #features = features.view(features.size(0), -1)
        #features = features.permute(0, 3, 2, 1)
        #features = self.proj(features)
        #features = features.permute(1, 0, 2, 3)
        features = features.permute(2,0,1)# 12 10 512
        return features



