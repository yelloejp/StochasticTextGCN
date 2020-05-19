#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np

class GraphConvolution(nn.Module):
    def __init__( self, input_dim, \
                        output_dim, \
                        support, \
                        act_func = None, \
                        featureless = False, \
                        dropout_rate = 0., \
                        bias=False):
        super(GraphConvolution, self).__init__()
        self.support = support
        self.featureless = featureless

        for i in range(len(self.support)):
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.randn(input_dim, output_dim)))

        if bias:
            self.b = nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.support)):
            if self.featureless:
                pre_sup = getattr(self, 'W{}'.format(i))
            else:
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))
            if i == 0:
                out = self.support[i].mm(pre_sup)                
            else:
                out += self.support[i].mm(pre_sup) # 거의 사용을 안하네??? 

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out


class GCN(nn.Module):
    def __init__( self, input_dim, \
                        support,\
                        dropout_rate=0., \
                        num_classes=10):
        super(GCN, self).__init__()
        
        # Need to build class for this part
        N, M = support[0].shape[0], support[0].shape[1]
        one = np.zeros(N*M).reshape(N,M)
        one[:, np.random.choice(one.shape[0], 30, replace=True)] = 1
        idx = np.random.rand(one.shape[0], one.shape[1]).argsort(axis=1)
        one = np.take_along_axis(one,idx,axis=1)
        support[0][:]=support[0]*one

        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, 200, support, act_func=nn.ReLU(), featureless=True, dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(200, num_classes, support, dropout_rate=dropout_rate)
        
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out
