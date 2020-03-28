# -*- coding: utf-8 -*-
"""
@author:
@date  :  2020/02/28
@desc:联邦学习单机版，host是带y标签一方，guest是不带y一方
"""
import sys
sys.path.append('../../')


import pandas as pd
import numpy as np
import warnings
import random
import math
import log
import os
import shutil
from phe import paillier
#过滤警告
warnings.filterwarnings(action='ignore')
#将日志输出到屏幕并保存到文件
logger = log.Logger(filename='./info.log').logger

def sampleAlignment(host_data,guest_data):
    """
    :param host_data: 带标签y的数据
    :param guest_data: 不带标签y的数据
    :return: 样本对齐输出uid列表
    """
    host_uid=np.asarray(host_data[['id']])
    guest_uid=np.asarray(guest_data[['id']])
    uid=np.intersect1d(host_uid,guest_uid)
    return uid

def encrypt_label(y):
    """
    对host数据中的label进行同态加密
    :param y: host dataset  DataFrame
    :return: 加密过后的label值，以及解密的私钥
    """
    # 对每一个y值添加非常微小的扰动
    epsilon = np.random.rand(len(y))
    epsilon = epsilon * 1e-10
    y = y + epsilon
    #对label进行加密
    publicKey,privacyKey=paillier.generate_paillier_keypair(n_length=128)

    encryptLabel=[]
    for val in y:
        encryptLabel.append(publicKey.encrypt(val))

    return np.asarray(encryptLabel), privacyKey

class BinomialDeviance:
    """二分类"""
    def initialize_f_0(self, y):
        """初始化F_0二分类的损失函数是对数损失，初始值是正样本的个数与负样本个数的比值取对数"""
        pos = y.sum()  #正样本
        neg = len(y) - pos  #负样本
        f_0_val = math.log(pos / neg)
        return np.ones(len(y))*f_0_val

    def gradients_(self,y_pred, y_true):
        """一阶导数 这里使用sigmoid函数的线性近似"""
        return 1 / (1 + np.exp(-y_pred)) - y_true

    def hessians_(self, y_pred, y_true=None):
        """二阶导数"""
        return np.exp(-y_pred) / ((1 + np.exp(-y_pred)) ** 2)

    def calculate_residual(self, y, current_value):
        """计算负梯度"""
        residual=-self.gradients_(current_value, y)
        return residual

    def update_f_m(self, X, current_value, trees, iter, learning_rate):
        """计算 当前时刻的预测值F_m ，"""
        for leaf_node in trees[iter].leaf_nodes:
            current_value[leaf_node.data_index]+=learning_rate*leaf_node.predict_value
        return current_value

    def update_leaf_values(self, targets, sample_weight=None):
        """更新叶子节点的预测值"""
        if len(targets)!=0:
            return (targets*sample_weight).sum()/sample_weight.sum()
        else:
            return 0.0

class EFB(object):
    def __init__(self,X,y=None):
        """
        互斥特征绑定 Exclusive Feature Bundling
        1，如何判断两个特征互斥：根据特征中的非0值判断两个特征的冲突度是多少,非零值越大，冲突越大
        2，两个互斥的特征怎么绑定：
        :param X:raw X
        :param y:raw y，虽然不对y进行处理
        """
        self.X=np.asarray(X)
        if y:
            self.y=np.asarray(y)
        self.bundles=None  #记录哪些特征是合并的
        self.findExclusiveFeature()
        self.dataset=self.bundlingFeature()

    def findExclusiveFeature(self):
        """
        寻找哪些特征是可以合并的
        :return: 多个保存合并特征的捆
        """
        Xcopy = self.X.copy()
        Xcopy[Xcopy != 0] = 1
        nonzeroCounts = np.sum(Xcopy, axis=0)  # [1,0,2]  每个特征非零值的数量
        feature_index = np.argsort(nonzeroCounts)  # 对这个数量进行一个排序，获得特征的索引，后面要用这个索引进行特征的合并
        # max_conflict_count = nonzeroCounts.sum() / len(nonzeroCounts)  # 用这些非零值的均值作为最大的冲突量
        max_conflict_count = 2
        bundles, bundle = [], {'index': [], 'conflict_count': 0}  # 前一个是所有捆的集合，后一个是放所有互斥的特征
        for i in range(len(feature_index)):
            index = feature_index[i]  # 当前特征索引，告诉我们是那个特征，是根据非零值从小到大排序的
            current_nonzeroCount = nonzeroCounts[index]  # 当前特征的非零值的个数
            if len(bundle['index']) == 0 and len(bundles) == 0:
                bundle['index'].append(index)
                bundle['conflict_count'] += current_nonzeroCount
                bundles.append(bundle)
            else:
                if bundles[-1]['conflict_count'] + current_nonzeroCount <= max_conflict_count:
                    # 如果加在现成的捆中总冲突量小于最大冲突量，则可以加到这个捆中
                    bundles[-1]['index'].append(index)
                    bundles[-1]['conflict_count'] += current_nonzeroCount
                else:
                    # 如果大于最大冲突量了，则新建一个捆
                    bundle = {'index': [], 'conflict_count': 0}
                    bundle['index'].append(index)
                    bundle['conflict_count'] += current_nonzeroCount
                    bundles.append(bundle)
        self.bundles=bundles

    def bundlingFeature(self):
        """
        对互斥的特征进行合并
        :return: 新的数据集
        """
        newSet = []
        for bundle in self.bundles:
            ef_index = bundle['index']  # 可以捆绑的特征
            sub_x = self.X[:, ef_index]
            max_value_per_fea = np.max(sub_x, axis=0)  # 每个特征的最大值，找一个偏移量
            offset = max_value_per_fea[0]
            if len(ef_index) > 1:
                a = self.X[:, ef_index[1:]]
                a[a != 0] += offset
                new_fea = np.add(a, sub_x[:, [ef_index[0]]])
                newSet.append(new_fea)
            else:
                newSet.append(sub_x)
        new_X = np.concatenate(newSet, axis=1)  # 生成新的特征
        return new_X

def goss(X, y_true, y_pred, loss=None, alpha=0.2, beta=0.1, random_state=20191127):
    """
    单边梯度采样
    :param X:
    :param y_true:lable 真实值
    :param y_pred:label 预测值
    :param loss:用到的损失
    :param alpha: 大梯度样本采样比例
    :param beta: 小梯度样本采样比例
    :param random_state: 随机种子
    :return:新数据以及样本权重
    """
    X,y_true,y_pred=np.asarray(X),np.asarray(y_true),np.asarray(y_pred)
    random.seed(random_state)
    # 计算所有样本的一阶导数和二阶导数，用的是对数损失函数
    n_samples=X.shape[0]
    grad_ = loss.gradients_(y_pred=y_pred,y_true=y_true)
    hess_ = loss.hessians_(y_pred=y_pred,y_true=y_true)
    # 对梯度进行排序
    gradientIndexSort = np.argsort(abs(grad_*hess_))  # 升序排列
    # 大梯度样本数量
    maxGrad = int(n_samples * alpha)
    # 因为是升序排列，所以取最后的maxGrad个样本
    num = n_samples - maxGrad
    topSet = gradientIndexSort[num:]  # 大梯度的值,这一部分的权重是不需要改变的
    # 小梯度样本，需要从剩下的样本中随机抽取b占比的样本吗，并修改这些样本的权重
    randSet = random.sample(list(gradientIndexSort[:num]), k=int(n_samples * beta))
    # 新的数据集合
    indexSort = np.concatenate([topSet, randSet], axis=0)
    indexSort = np.asarray(indexSort,dtype=np.int32)
    # 更新权重值
    sample_weight = np.ones(indexSort.shape[0])
    sample_weight[int(n_samples * alpha):] *= (1 - alpha) / beta
    return indexSort, sample_weight

def histogram(X,gradient,role='host',max_bin=255,min_data_bin=1):
    n_sampels,n_features=X.shape[0],X.shape[1]
    histogram_dict={}
    G_all = gradient.sum()  #这个值也是要传给guest的
    for i in range(n_features):
        key=role+'_'+str(i)
        histogram_dict[key]=histogram_dict.get(key,None)
        H={}
        for j in range(n_sampels):
            H[X[j][i]]=H.get(X[j][i],[0,0])
            H[X[j][i]][0]+=gradient[j]
            H[X[j][i]][1]+=1

        binSet = sorted(H.items(), key=lambda x: x[0], reverse=False)
        binSet = [b for b in binSet if b[1][1] >= min_data_bin][:max_bin]
        histogram_dict[key]=binSet
        # sl, nl = 0, 0  # 当前桶左边的梯度之和与样本数量
        # for k in range(len(binSet)):
        #     sl += binSet[k][1][0]
        #     nl += binSet[k][1][1]
        #     sr = G_all - sl
        #     nr = n_sampels - nl
        #     # 计算当前节点的信息增益
        #     if nl == 0 or nr == 0:
        #         infoGainTemp = 0
        #     else:
        #         infoGainTemp = sl ** 2 / nl + sr ** 2 / nr - G_all ** 2 / n_sampels
        #     if host:
        #         key='host_'+str(i)
        #     else:
        #         key='guest_'+str(i)
        #     if infoGainTemp>histogram_dict[key]['infoGain']:
        #         histogram_dict[key]['infoGain']=infoGainTemp
        #         histogram_dict[key]['best_split_value']=X[j][i]

    return histogram_dict

class Node:
    """创建一个树的节点"""
    def __init__(self,
                 data_index,
                 split_feature=None,
                 split_value=None,
                 is_leaf=False,
                 loss=None,
                 current_depth=None):
        """
        :param data_index:该节点的数据在全部数据集中的索引
        :param split_feature: 最佳分割特征
        :param split_value:最佳分割特征值
        :param is_leaf:是否为叶子节点
        :param loss: 分类损失
        :param current_depth: 当前节点所在树的深度
        """
        self.loss = loss
        self.split_feature = split_feature
        self.split_value = split_value
        self.data_index = data_index
        self.is_leaf = is_leaf  #是不是叶子节点
        self.predict_value = None
        self.left_child = None  #如果当前节点可继续划分，则分为左子树和右子树
        self.right_child = None
        self.current_depth = current_depth

    def update_predict_value(self, targets, sample_weight=None):
        self.predict_value = self.loss.update_leaf_values(targets, sample_weight=sample_weight)
        logger.info('>>>>>>>>>>>叶子节点预测值：%.3f'%self.predict_value)

    def get_predict_value(self, instance):
        """
        预测结果，采用递归的方法
        :param instance: 一个样本
        :return:
        """
        if self.is_leaf:
            #如果是叶子节点，直接获得叶子节点值
            return self.predict_value
        if instance[self.split_feature] < self.split_value:
            return self.left_child.get_predict_value(instance)
        else:
            return self.right_child.get_predict_value(instance)

class Tree:
    """创建一棵树，从根节点开始创建"""
    def __init__(self,
                 host_X,
                 guest_X,
                 gradient,
                 encrypt_gradient,
                 privacyKey,
                 current_tree,
                 max_depth=3,
                 min_samples_split=2,
                 features_name=None,
                 loss=None,
                 sample_weight=None,
                 max_bin=255,
                 min_data_bin=1):
        """
        初始化树的参数
        :param host_X:
        :param gradient:
        :param max_depth:树的最大深度
        :param min_samples_split:最小划分数据量
        :param sample_weight: 样本权重
        :param features_name:特征名称
        :param loss:
        :param max_bin:桶的最大数量
        :param min_data_bin:桶内最小样本量，目前测试阶段设置为1
        :return: 树结构和节点
        """
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_weight=np.ones(len(host_X)) if sample_weight is None else sample_weight
        self.features_name = features_name
        self.privacyKey=privacyKey
        self.remain_index = np.array([i for i in range(len(host_X))])  #当前节点的样本在原始数据中的下标索引
        self.leaf_nodes = []
        self.max_bin=max_bin
        self.min_data_bin=min_data_bin
        self.root_node = self.build_tree(host_X, guest_X, gradient, encrypt_gradient, current_tree, self.remain_index, depth=0, sample_weight=self.sample_weight)  #根节点

    def build_tree(self, host_X, guest_X, gradient, encrypt_gradient, current_tree, remain_index, depth=0, sample_weight=None):
        """
        构建一棵树
        此处有三个树继续生长的条件：
        1: 深度没有到达最大, 树的深度假如是3， 意思是需要生长成3层, 那么这里的depth只能是0, 1，所以判断条件是 depth < self.max_depth - 1
        2: 点样本数 >= min_samples_split
        3: 此节点上的样本的 target_name 值不一样（如果值 一样说明已经划分得很好了，不需要再分）
        :param current_tree: 当前值
        :param sample_weight:样本权重
        :param X: 原始样本特征值
        :param y: 原始样本标签值
        :param remain_index:当前数据在原始数据中的索引
        :param depth: 树的当前深度
        :return: 树节点
        """
        now_host_X, now_gradient, now_current_tree = host_X[remain_index], gradient[remain_index], current_tree[remain_index]
        now_guest_X=guest_X[remain_index]
        n_samples, n_features=now_host_X.shape[0], now_host_X.shape[1]

        logger.info('----------host构建直方图----------')
        host_histogram=histogram(now_host_X,gradient,'host')
        logger.info('----------guest构建直方图----------')
        guest_histogram=histogram(now_guest_X,encrypt_gradient,'guest')
        logger.info('----------guest将直方图传给host----------')

        if depth < self.max_depth and n_samples >= self.min_samples_split and len(np.unique(now_current_tree)) > 1:
            split_feature = None
            split_value = None
            maxInfoGain=0
            G_all=gradient.sum()
            logger.info('----------host计算最优划分点和划分值----------')
            #TODO 确定最优点和最优特征 分桶优化后这个地方要更改
            for feature in host_histogram:
                binSet=host_histogram[feature]
                sl, nl = 0, 0  # 当前桶左边的梯度之和与样本数量
                for k in range(len(binSet)):
                    sl += binSet[k][1][0]
                    nl += binSet[k][1][1]
                    sr = G_all - sl
                    nr = n_samples - nl
                    # 计算当前节点的信息增益
                    if nl == 0 or nr == 0:
                        infoGainTemp = 0
                    else:
                        infoGainTemp = sl ** 2 / nl + sr ** 2 / nr - G_all ** 2 / n_samples
                    if infoGainTemp>maxInfoGain:
                        maxInfoGain=infoGainTemp
                        split_feature=feature
                        split_value=binSet[k][0]

            for feature in guest_histogram:
                binSet = guest_histogram[feature]
                sl, nl = 0, 0  # 当前桶左边的梯度之和与样本数量
                for k in range(len(binSet)):
                    sl += round(privacyKey.decrypt(binSet[k][1][0]),1)  #解密呀斯密达
                    nl += binSet[k][1][1]
                    sr = G_all - sl
                    nr = n_samples - nl
                    # 计算当前节点的信息增益
                    if nl == 0 or nr == 0:
                        infoGainTemp = 0
                    else:
                        infoGainTemp = sl ** 2 / nl + sr ** 2 / nr - G_all ** 2 / n_samples
                    if infoGainTemp > maxInfoGain:
                        maxInfoGain = infoGainTemp
                        split_feature = feature
                        split_value = binSet[k][0]
            #TODO host将计算的最佳分割特征和分割值传给guest
            #判断这个特征属于host还是guest,如果是host则host计算分割后的节点数据传给guest，反之。

            role,index=split_feature.split('_')[0],split_feature.split('_')[1]
            if role=='host':
                logger.info('----------最优划分特征属于host----------')
                left_index = list(now_host_X[:, int(index)] <= split_value)
                right_index = list(now_host_X[:, int(index)] > split_value)
                logger.info('----------host将分割样本索引传给guest----------')
            else:
                logger.info('----------最优划分特征属于guest----------')
                logger.info('----------host将最优划分点和划分值传给guest----------')
                left_index = list(now_guest_X[:, int(index)] <= split_value)
                right_index = list(now_guest_X[:, int(index)] > split_value)
                logger.info('----------guest将分割样本索引传给host----------')
            left_index_of_now_data = remain_index[left_index]
            right_index_of_now_data = remain_index[right_index]
            left_weight_index_of_now_data = left_index
            right_weight_index_of_now_data = right_index

            logger.info('----------最佳划分特征：%s----------'%split_feature)
            logger.info('----------最佳划分值：%.3f----------'%split_value)

            #找到当前最优划分特征和划分值，对当前这个节点进行处理，产生子节点
            node = Node(remain_index, split_feature, split_value, current_depth=depth)

            logger.info('----------构建左子树----------')
            node.left_child = self.build_tree(host_X, guest_X, gradient,encrypt_gradient, current_tree, left_index_of_now_data, depth=depth + 1, sample_weight=sample_weight[left_weight_index_of_now_data])
            logger.info('----------构建右子树----------')
            node.right_child = self.build_tree(host_X, guest_X, gradient,encrypt_gradient, current_tree, right_index_of_now_data, depth=depth + 1, sample_weight=sample_weight[right_weight_index_of_now_data])
            return node
        else:
            #不满足分裂条件，停止分裂，该节点就是叶子节点
            logger.info('----------树的深度：%d----------' % depth)
            node = Node(remain_index, is_leaf=True, loss=self.loss, current_depth=depth)
            node.update_predict_value(current_tree[remain_index], sample_weight=sample_weight)
            self.leaf_nodes.append(node)
            return node


if __name__ == '__main__':

    # 创建模型结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    if len(os.listdir('results')) > 0:
        shutil.rmtree('results')
        os.makedirs('results')

    #加载数据
    logger.info('---------加载数据---------')
    logger.info('---------加载host方数据---------')
    host_data=pd.read_csv('../demo/breast_b.csv')   #带label
    logger.info('---------加载guest方数据---------')
    guest_data=pd.read_csv('../demo/breast_a.csv')  #不带label
    logger.info('----------host和guest方数据样本对齐----[开始]')
    uid=sampleAlignment(host_data,guest_data)

    host_y=np.asarray(host_data['y'][uid])
    host_x = host_data.iloc[uid, 2:]
    guest_x = guest_data.iloc[uid, 1:]
    logger.info('----------host和guest方数据样本对齐----[完成]')

    #host和guest各自进行EFB
    logger.info('----------开始互斥特征合并----------')

    logger.info('----------host特征合并----------')
    hostEFB=EFB(host_x)
    host_x=np.asarray(hostEFB.dataset)
    hostBundles=hostEFB.bundles

    logger.info('----------guest特征合并----------')
    guestEFB=EFB(guest_x)
    guest_x=np.asarray(guestEFB.dataset)
    guestBundles=guestEFB.bundles

    #计算初始值 lightgbm中初始值为0
    init_value=np.zeros(len(uid))
    #定义损失
    loss=BinomialDeviance()
    current_value=init_value
    #先建一棵树
    n_estimators=4
    trees=[]
    """进行单边梯度采样,采样之后的样本权重和uid需要传给guest"""
    logger.info('======================分割线=========================')
    for iter in range(n_estimators):
        logger.info('===>>>-*START*- LightGBM Model Training ')
        logger.info('---------------->>>>>>  构建第%d颗树  <<<<<<----------------' % (iter+1))
        sample_data_index, sample_weight = goss(X=host_x,y_true=host_y,y_pred=current_value,loss=loss)
        if len(sample_data_index)==0:
            logger.info("----------单边采样数据量为0，训练结束----------")
            break
        current_value=current_value[sample_data_index]
        logger.info('----------host将采样后的uid传给guest----------')
        #对于host
        host_x, host_y = host_x[sample_data_index], host_y[sample_data_index]
        #对于guest，利用host传过来的uid，获得新的样本集
        guest_x = guest_x[sample_data_index]

        residual = loss.calculate_residual(host_y,current_value=current_value)  # 计算负梯度
        """计算uid对应的一阶梯度，这个要加密给guest"""
        grad_uid=-residual
        encrypt_gradient,privacyKey=encrypt_label(grad_uid)
        logger.info('----------host将【加密后的梯度信息】传给guest----------')

        """这个树交给host来创建"""
        tree=Tree(host_X=host_x,
                  guest_X=guest_x,
                  current_tree=residual,
                  gradient=grad_uid,
                  encrypt_gradient=encrypt_gradient,
                  privacyKey=privacyKey,
                  max_depth=3,
                  min_samples_split=2,
                  features_name=None,
                  loss=loss,
                  sample_weight=sample_weight,
                  max_bin=255,
                  min_data_bin=1)
        trees.append(tree)
        current_value=loss.update_f_m(host_x, current_value, trees, iter, learning_rate=0.1)

    logger.info('=================< -*END*- LightGBM Model Training >=================')






