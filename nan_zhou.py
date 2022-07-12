'''
Author: Zhou Hao
Date: 2021-11-02 17:14:04
LastEditors: ZhouHao 2294776770@qq.com
LastEditTime: 2022-07-12 10:47:05
Description: file content
E-mail: 2294776770@qq.com
'''
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import numpy as np 
import math


class Natural_Neighbor(object):

    def __init__(self,X:np.array,y:np.array):
        self.nan_edges = {}        # Graph of mutual neighbors
        self.nan_num = {}          # Number of natural neighbors of each instance
        self.target:np.array = y           # Set of classes
        self.data:np.array = X             # Instance set
        self.knn = {}              # Structure that stores the neighbors of each instance
        self.nan = {}           # Structure that stores the nan neighbors of each instance
        self.relative_cox = []
        

    def asserts(self):
        self.nan_edges = set()
        for j in range(len(self.data)):
            self.knn[j] = set()
            self.nan[j] = set()     # 初始化自然邻居
            self.nan_num[j] = 0

    # Returns the number of instances that have no natural neighbor
    def count(self):
        nan_zeros = 0
        for x in self.nan_num:
            if self.nan_num[x] == 0:
                nan_zeros += 1
        return nan_zeros

    # Returns the indices of the closest neighbors
    def findKNN(self, inst, r, tree):
        _dist, ind = tree.query([inst], r+1)  # r+1最近邻的距离, 下标
        return np.delete(ind[0], 0)

    # 返回 NaNe
    def algorithm(self):
        # ASSERT
        tree = KDTree(self.data)    # 构建kD树
        self.asserts()              # 初始化计算自然邻居参数:self.knn,self.nan_num,self.nan_edges
        flag = 0
        r = 2   # 自然特征初始化为1
        cnt_before = -1  # 初始化当前邻居数为0的样本总数为负数

        while(flag == 0 and r <= 5)  :
            for i in range(len(self.data)):
                knn = self.findKNN(self.data[i], r, tree)
                n = knn[-1]  # 获取r-最近邻
                self.knn[i].add(n)  # 添加i的r-最近邻n
                if(i in self.knn[n] and (i, n) not in self.nan_edges):  # 判断自然邻居
                    self.nan_edges.add((i, n))
                    self.nan_edges.add((n, i))
                    self.nan[i].add(n)      # 记录自然邻居对
                    self.nan[n].add(i)
                    self.nan_num[i] += 1    # i,n的自然邻居数加一
                    self.nan_num[n] += 1

            cnt_after = self.count()  # 获取当前邻居数为0的样本总数
            if cnt_after < math.sqrt(len(self.data)):
                flag = 1
            else:
                r += 1
            cnt_before = cnt_after
        return r,tree


    # 相对密度
    def RelativeDensity(self, min_i, maj_i):
        """离群点密度-2,噪声点-3,多数类0"""
        self.relative_cox = [0]*len(self.target)
        for i, num in self.nan.items():   # num为i的自然邻居集合

            if self.target[i] == min_i:     #只计算少数类
                if len(num) == 0 :  # 离群样本(没有自然邻居):-2 
                    self.relative_cox[i] = -2
                else:
                    absolute_min,min_num,absolute_max,maj_num = 0,0,0,0
                    maj_index = []

                    for j in iter(num): # 遍历自然邻居集合
                        if self.target[j] == min_i:     # 自然邻居是少数类
                            absolute_min += np.sqrt(np.sum(np.square(self.data[i]-self.data[j])))
                            min_num += 1
                        elif self.target[j] == maj_i:   # 自然邻居是多数类
                            absolute_max += np.sqrt(np.sum(np.square(self.data[i]-self.data[j])))
                            maj_num += 1
                            maj_index.append(j)
                    self.nan[i].difference_update(maj_index)    #去掉自然邻居中的多数类样本

                    # 计算当前样本的密度权重
                    if min_num == 0:    # 噪声点(自然邻居全是多数类) -3: 过滤少数类(删除噪声点)
                        self.relative_cox[i] = -3 
                    elif maj_num == 0 :         # 安全点: 自然邻居都为少数类
                        relative = min_num/absolute_min
                        self.relative_cox[i] = relative
                    else:           # 边界样本:   自然邻居中有多数类也有少数类
                        relative = (min_num/absolute_min)/(maj_num/absolute_max)
                        self.relative_cox[i] = relative