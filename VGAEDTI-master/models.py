import torch
import torch.nn as nn
import torch.nn.functional as F

# 继承torch.nn.module
class GraphConv(nn.Module):
    def __init__(self,in_dim, out_dim, drop=0.5, bias=False, activation=None): # self实例本身，构造函数
# 首先找到GraphConv的父类（比如是类A），然后把类GraphConv的对象self转换为类A的对象，然后“被转换”的类A对象调用自己的__init__函数
        super(GraphConv,self).__init__() # 对继承自父类的属性进行初始化
        self.dropout = nn.Dropout(drop) # 表示为GraphConv类添加了属性dropout
        self.activation = activation
        self.w = nn.Linear(in_dim, out_dim, bias=bias) # y=wx+b
        nn.init.xavier_uniform_(self.w.weight) # 为了训练过程中前后的方差稳定问题，正确的初始化有利于训练的稳定；
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias) # 参数初始化， 用标量值 0 填充偏执。
    # 前向传播
    def forward(self, adj, x):
        x = self.dropout(x) # dropout训练
        x = adj.mm(x) # adj与x相乘
        x = self.w(x) # 上面的self.w方法是y=wx+b
        if self.activation: # 如果使用激活函数返回激活后的值，不使用就直接返回
            return self.activation(x)
        else:
            return x
# 变分图自动编码器
class AE(nn.Module):
    def __init__(self,feat_dim,hid_dim,out_dim,bias=False):
        super(AE,self).__init__()
        self.conv1 = GraphConv(feat_dim,hid_dim,bias=bias,activation=F.relu)
        self.mu = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)
        self.conv3 = GraphConv(out_dim,hid_dim,bias=bias,activation=F.relu)
        self.conv4 = GraphConv(hid_dim,feat_dim,bias=bias,activation=torch.sigmoid)
        self.logvar = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)

    def encoder(self,g,x):
        x = self.conv1(g,x)
        h = self.mu(g,x)
        std = self.logvar(g,x)
        return h,std
    
    def decoder(self,g,x):
        x = self.conv3(g,x)
        x = self.conv4(g,x)
        return x
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar) # 指数函数
            eps = torch.randn_like(std) # 返回一个和输入大小相同的张量，其由均值为0、方差为1的标准正态分布填充。
            return eps.mul(std).add_(mu) # mul()相乘，原数不变，mul_()相乘，原数变动，add()同理
        else:
            return mu
    # 更新参数后的前向传播
    def forward(self,g,x):
        mu,logvar = self.encoder(g,x) # 重新进行编码
        z = self.reparameterize(mu, logvar) # 参数更新
        return mu,logvar,self.decoder(g,z)
# 实现标签传播算法(LP算法)
class LP(nn.Module):
    def __init__(self,hid_dim,out_dim,bias=False):
        super(LP,self).__init__()
        self.res1 = GraphConv(out_dim,hid_dim,bias=bias,activation=F.relu)
        self.res2 = GraphConv(hid_dim,out_dim,bias=bias,activation=torch.sigmoid)
    
    def forward(self,g,z):
        z = self.res1(g,z)
        res = self.res2(g,z)
        return res,z
