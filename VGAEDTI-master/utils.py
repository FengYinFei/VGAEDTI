import numpy as np
import torch
import argparse
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale,scale
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc

def scaley(ymat):
    return (ymat-ymat.min())/ymat.max() # 数据的规范化

# 在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果，
# 当得到比较好的结果时我们通常希望这个结果是可以复现的，在pytorch中，通过设置随机数种子可以达到这个目的。
def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    if cuda:
        torch.cuda.manual_seed(seed) # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
# 加载数据
def load_data(data,cuda):
    path = 'Dataset'+str(data)
    gdi = np.loadtxt(path + '/test-disease-drug.txt')
    ldi = np.loadtxt(path + '/test-protein-drug.txt')
    drugfeat = np.loadtxt(path + '/test-Similarity_Drug.txt', delimiter=',')
    proteinfeat = np.loadtxt(path + '/test-Similarity_Proteins.txt', delimiter=',')
    proteinfeat = minmax_scale(proteinfeat,axis=0)# 数据归一化（默认范围0-1）
    gdit = torch.from_numpy(gdi).float()# numpy转换为张量
    ldit = torch.from_numpy(ldi).float()
    drugit = torch.from_numpy(drugfeat).float()
    proteinfeatorch = torch.from_numpy(proteinfeat).float()
    gl = norm_adj(proteinfeat) #调用下面对数据处理的方法
    gd = norm_adj(gdi.T)

    # cuda()加载gpu
    if cuda:
        gdit = gdit.cuda()
        ldit = ldit.cuda()
        proteinfeatorch = proteinfeatorch.cuda()
        gl = gl.cuda()
        gd = gd.cuda()
    
    return gdit, ldit, proteinfeatorch, gl, gd

def neighborhood(feat,k):
    # compute C
    featprod = np.dot(feat.T,feat)# 两矩阵点积运算
    # np.tile将函数将函数沿着X轴扩大两倍。如果扩大倍数只有一个，默认为X轴[1 ,2,3]扩大后[1,2,3,1,2,3]
    # np.tile(a, (2, 1))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数。
    # np.diag array是一个1维数组时，结果形成一个以一维数组为对角线元素的矩阵
    # array是一个二维矩阵时，结果输出矩阵的对角线元素
    # shape.[0or1],0表示行，1表示列
    smat = np.tile(np.diag(featprod),(feat.shape[1],1)) # title(上面矩阵乘积运算后的对角线元素，（输入元素的形状的列,1）)
    dmat = smat + smat.T - 2*featprod # 矩阵的加减运算
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    # 二维数组排序,[:]复制原数组，[1:k+1]输出索引值1到k+1
    dsort = np.argsort(dmat)[:,1:k+1]
    # np.zeros函数的作用返回来一个给定形状和类型的用0填充的数组
    C = np.zeros((feat.shape[1],feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i,j] = 1.0
    
    return C

def normalized(wmat):
    # np.sum(a, axis=0) - ------>列求和
    # np.sum(a, axis=1) - ------>行求和
    deg = np.diag(np.sum(wmat,axis=0)) # 求和后取出对角线元素
    # np.power()用于数组元素求n次方
    # result = np.power(x1,x2) # 实际就是相应位置的前者的后者次方(x1[i,j]**x2[i,j])
    degpow = np.power(deg,-0.5)
    # 用于检查数字是否为无穷大(正数或负数)，它接受数字，如果给定数字为正无穷大或负无穷大，则返回True。返回False 。
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    return W

def norm_adj(feat):
    C = neighborhood(feat.T,k=10)
    # np.eye返回的是一个二维2的数组(N,M)，对角线的地方为1，其余的地方为0.
    norm_adj = normalized(C.T*C+np.eye(C.shape[0]))# 做一个加法乘法运算
    g = torch.from_numpy(norm_adj).float()
    return g

def show_auc(ymat,data):
    path = 'Dataset'+str(data)
    ldi = np.loadtxt(path + '/test-protein-drug.txt')
    y_true = ldi.flatten()
    ymat = ymat.flatten()
    fpr,tpr,rocth = roc_curve(y_true,ymat)
    auroc = auc(fpr,tpr)
    np.savetxt('roc.txt',np.vstack((fpr,tpr)),fmt='%10.5f',delimiter=',')
    precision,recall,prth = precision_recall_curve(y_true,ymat)
    np.savetxt('E:/代码参考/VGAEDTI/VGAEDTI-master/recall1.txt',recall,fmt='%10.2f',delimiter=',')
    aupr = auc(recall,precision)
    np.savetxt('pr.txt',np.vstack((recall,precision)),fmt='%10.5f',delimiter=',')
    print('AUROC= %.4f | AUPR= %.4f' % (auroc,aupr))
    rocdata = np.loadtxt('roc.txt',delimiter=',')
    prdata = np.loadtxt('pr.txt',delimiter=',')
    # print('ROC= %.4f | PR= %.4f' % (rocdata, prdata))
    plt.figure()
    plt.plot(rocdata[0],rocdata[1])
    plt.plot(prdata[0],prdata[1])
    plt.show()
    return auroc, aupr