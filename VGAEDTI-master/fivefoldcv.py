import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from models import GraphConv, AE, LP
from utils import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# from Randomforest import randomforest
# 创建一个解析器——创建 ArgumentParser() 对象
parser = argparse.ArgumentParser()
# 添加参数——调用 add_argument() 方法添加参数
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-8,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight between drug space and protein space')
parser.add_argument('--data', type=int, default=1, choices=[1,2],
                    help='Dataset')
# 添加参数——调用 add_argument() 方法添加参数
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# set.seed()产生随机数
set_seed(args.seed,args.cuda)
gdi, ldi, proteinfeat, gl, gd = load_data(args.data,args.cuda)
# 变分图自动编码器
class GNNq(nn.Module):
    def __init__(self):
        super(GNNq,self).__init__()
        self.gnnql = AE(proteinfeat.shape[1],256,args.hidden)
        self.gnnqd = AE(gdi.shape[0],256,args.hidden)
    
    def forward(self,xl0,xd0):
        hl,stdl,xl = self.gnnql(gl,xl0)
        hd,stdd,xd = self.gnnqd(gd,xd0)
        return hl,stdl,xl,hd,stdd,xd
# 图自动编码器
class GNNp(nn.Module):
    def __init__(self):
        super(GNNp,self).__init__()
        self.gnnpl = LP(args.hidden,ldi.shape[1])
        self.gnnpd = LP(args.hidden,ldi.shape[0])

    def forward(self,y0):
        yl,zl = self.gnnpl(gl,y0)
        yd,zd = self.gnnpd(gd,y0.t())
        return yl,zl,yd,zd

print("Dataset {}, 5-fold CV".format(args.data))

# 损失函数的使用
def criterion(output,target,msg,n_nodes,mu,logvar):
    if msg == 'drug':
        cost = F.binary_cross_entropy(output,target) # 交叉熵损失函数
    else:
        cost = F.mse_loss(output,target) # 均方误差损失函数
    # kl散度：描述两个概率分布P和Q差异的一种方法，p真实分布，q表示的拟合分布
    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL
# 训练阶段
def train(gnnq,gnnp,xl0,xd0,y0,epoch,alpha):
    beta0 = 0.0001
    gamma0 = 0.0001
    lossf = 0.1
    # 优化器，学习率，权重衰减
    optp = torch.optim.Adam(gnnp.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    optq = torch.optim.Adam(gnnq.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    for e in range(epoch):
        gnnq.train()
        hl,stdl,xl,hd,stdd,xd = gnnq(xl0,xd0)
        lossql = criterion(xl,xl0,
            "protein",gl.shape[0],hl,stdl)
        lossqd = criterion(xd,xd0,
            "drug",gd.shape[0],hd,stdd)
        lossq = alpha*lossql + (1-alpha)*lossqd + beta0*e*F.mse_loss(
            torch.mm(hl,hd.t()),y0)/epoch
        optq.zero_grad() # 意思是把梯度置零，也就是把loss关于weight的导数变成0.理解成一种梯度下降法
        lossq.backward() # 反向传播求梯度
        optq.step() # 更新所有参数
        gnnq.eval() # eval() 函数用来执行一个字符串表达式，并返回表达式的值。相当于输出语句
        with torch.no_grad(): # 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
            hl,_,_,hd,_,_ = gnnq(xl0,xd0)
        
        gnnp.train()
        yl,zl,yd,zd = gnnp(y0)
        losspl = F.binary_cross_entropy(yl,y0) + gamma0*e*F.mse_loss(zl,hl)/epoch
        losspd = F.binary_cross_entropy(yd,y0.t()) + gamma0*e*F.mse_loss(zd,hd)/epoch

        lossp = alpha*losspl + (1-alpha)*losspd
       # lossp = alpha * losspl * lossf + (1 - alpha) * losspd * lossf

        optp.zero_grad()
        lossp.backward()
        optp.step()

        gnnp.eval()
        with torch.no_grad():
            yl,_,yd,_ = gnnp(y0)
        
        if e%20 == 0:
            print('Epoch %d | GAELoss: %.4f | VGAELoss: %.4f' % (e, lossp.item(),lossq.item()))
        
    return alpha*yl+(1-alpha)*yd.t()
# 五倍交叉验证
def fivefoldcv(A,alpha=0.5):
    N = A.shape[0] # 行数
    idx = np.arange(N) # 返回行的个数
    np.random.shuffle(idx) # shuffle() 方法将序列的所有元素随机排序。
    res = torch.zeros(5,A.shape[0],A.shape[1]) # 输出一个5列的A行A列个数的tensor
    aurocl = np.zeros(5) # 输出5列的0
    auprl = np.zeros(5)
    for i in range(5):
        print("Fold {}".format(i+1))
        A0 = A.clone() # 张量的复制操作
        for j in range(i*N//5,(i+1)*N//5):
            A0[idx[j],:] = torch.zeros(A.shape[1])
        
        gnnq = GNNq()
        gnnp = GNNp()
        if args.cuda:
            gnnq = gnnq.cuda()
            gnnp = gnnp.cuda()

        train(gnnq,gnnp,proteinfeat,gdi.t(),A0,args.epochs,args.alpha)
        gnnq.eval()
        gnnp.eval()
        yli,_,ydi,_ = gnnp(A0)
        resi = alpha*yli + (1-alpha)*ydi.t()
        #resi = scaley(resi)
        res[i] = resi
        
        if args.cuda:
            resi = resi.cpu().detach().numpy()
        else:
            resi = resi.detach().numpy()
        
        auroc,aupr = show_auc(resi,args.data)
        aurocl[i] = auroc
        auprl[i] = aupr
        
    ymat = res[auprl.argmax()]
    if args.cuda:
        return ymat.cpu().detach().numpy()
    else:
        return ymat.detach().numpy()


def randomforest():

    mtrain, dtrain = load_data(1,cuda=2)

    encoder, m_data1 = GNNq(mtrain)
    encoder, d_data1 = GNNp(dtrain)
    num_cross = 5
    probaresult = []
    ae_y_pred_probresult = []

    for fold in range(num_cross):
        train_m = np.array([x for i, x in enumerate(m_data1) if i % num_cross != fold])
        test_m = np.array([x for i, x in enumerate(m_data1) if i % num_cross == fold])
        train_d = np.array([x for i, x in enumerate(d_data1) if i % num_cross != fold])
        test_d = np.array([x for i, x in enumerate(d_data1) if i % num_cross == fold])
        train_label = np.array([x for i, x in enumerate(m_data1) if i % num_cross != fold])

        train_label_new = []

        for val in train_label:
            if val[0] == 1:
                train_label_new.append(0)
            else:
                train_label_new.append(1)

        prefilter_mtrain = train_m
        prefilter_mtest = test_m
        prefilter_dtrain = train_d
        prefilter_dtest = test_d

        # 随机森林分类器
        # drug
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(prefilter_mtrain, train_label_new)# clf.fit用训练数据拟合分类器模型
        # clf.predict_proba返回预测属于某标签的概率,是取二维数组中第二维的所有数据
        mae_y_pred_prob = clf.predict_proba(prefilter_mtest)[:, 1]
        # 调用下面的函数如果大于0.5为1，否则0
        mproba = transfer_label_from_prob(mae_y_pred_prob)
        # protein
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(prefilter_dtrain, train_label_new)
        dae_y_pred_prob = clf.predict_proba(prefilter_dtest)[:, 1]
        dproba = transfer_label_from_prob(dae_y_pred_prob)

        mproba = np.array(mproba)
        dproba = np.array(dproba)

        proba = (mproba + dproba)/2
        ae_y_pred_prob = (mae_y_pred_prob + dae_y_pred_prob)/2

        probaresult.extend(proba)
        ae_y_pred_probresult.extend(ae_y_pred_prob)
    return probaresult, ae_y_pred_probresult, m_data1
def Randomforest():
    probafs = []
    ae_y_pred_probafs = []
    lables = []
    return probafs, ae_y_pred_probafs, lables

def transfer_label_from_prob(proba):
    # 将像素指定为0和1
    label = [1 if val >= 0.5 else 0 for val in proba]
    return label
probafs = []
ae_y_pred_probafs = []
probaresult, ae_y_pred_probresult, lables = Randomforest() # 调用随机森林分类器
probafs.extend(probaresult)
ae_y_pred_probafs.extend(ae_y_pred_probresult)
probafs = np.array(probafs)
ae_y_pred_probafs = np.array(ae_y_pred_probafs)
'''probafs = []
ae_y_pred_probafs = []
for i in range(1):
 probaresult, ae_y_pred_probresult, labels = randomforest()# 调用随机森林分类器
''''''
probafs.extend(probaresult)
ae_y_pred_probafs.extend(ae_y_pred_probresult)
probafs = np.array(probafs)
ae_y_pred_probafs = np.array(ae_y_pred_probafs)
'''
title = 'result-test-dataset'+str(args.data)
ymat = fivefoldcv(ldi,alpha=args.alpha)
title += '--fivefoldcv'
ymat = scaley(ymat)
np.savetxt(title+'.txt',ymat,fmt='%10.5f',delimiter=',')
print("===Final result===")
show_auc(ymat,args.data)