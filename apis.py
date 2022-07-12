'''
Author: Zhou Hao
Date: 2022-01-20 20:26:28
LastEditors: ZhouHao 2294776770@qq.com
LastEditTime: 2022-07-12 10:07:12
Description: apis of smote
E-mail: 2294776770@qq.com
'''

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import random
from random import choice
from sklearn.datasets import make_circles, make_moons,make_swiss_roll
from sklearn.cluster import KMeans


def swiss_roll(n_samples=2000,noise=0.3) ->np.array:
    """三维二分类数据集"""
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise)
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    y_0 = np.where(y_pred == 0)[0]
    y_1 = np.where(y_pred == 1)[0]
    y_0 = np.random.choice(y_0,200)
    y_1 = np.random.choice(y_1,1000)

    data = np.hstack((y_pred.reshape(X.shape[0],1),X))  # 标签是第一列
    data_ = np.vstack((data[y_0],data[y_1]))

    X,y = data_[:,1:],data_[:,0] 

    # 添加翻转噪声
    index = random.sample( list(range(len(y))), int(len(y)*0.05))
    for i in index:
        if y[i] == 0: y[i] = 1
        else:y[i] = 0

    dataset = np.hstack([y.reshape(X.shape[0],1),X])    # 标签是第一列
    train = add_flip_noise(dataset,0)  
    X,y = train[:,1:],train[:,0] 
    return X,y


def add_flip_noise(datasets:np.ndarray, noise_rate:float)->np.ndarray:

    label_cat = list(set(datasets[:, 0]))
    new_data = np.array([])
    flag = 0
    for i in range(len(label_cat)):
        label = label_cat[i]
        other_label = list(filter(lambda x: x != label, label_cat))
        data = datasets[datasets[:, 0] == label]
        n = data.shape[0]
        noise_num = int(n * noise_rate)
        noise_index_list = []  # 记录所有噪声的下标
        n_index = 0
        while True:
            # 每次选择下标
            rand_index = int(random.uniform(0, n))
            # 如果下标已有，执行下一次while
            if rand_index in noise_index_list:
                continue
            # 满足两个条件翻转: 正类且噪声噪声不够
            if n_index < noise_num:
                data[rand_index, 0] = choice(other_label)  # todo
                n_index += 1
                noise_index_list.append(rand_index)
            # 跳出
            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = np.vstack([new_data, data])
    return new_data


def binary_data(data_name:str, random_state=42)->np.ndarray:

    np.random.seed(random_state)
    if data_name == 'make_moons':
        x, y = make_moons(n_samples=2000, noise=0.30)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)
        
        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]
        data_0 = data_0[:250]
        data = np.vstack((data_0, data_1))
        
    elif data_name == 'make_circles':
        x, y = make_circles(n_samples=600, noise=0.3, factor=0.2)
        data = np.hstack((y.reshape((len(y), 1)), x))
        np.random.shuffle(data)

        data_0 = data[data[:, 0] == 0]
        data_1 = data[data[:, 0] == 1]
        data_0 = data_0[:30]
        data = np.vstack((data_0, data_1))

    data = pd.DataFrame(data)
    y = data[0]
    X = data[[1, 2]]
    X = X.values
    y = y.values
    
    return X, y


def multi_class_data(dataset_name:str,noise_rate,n_samples=1500,n_clusters=3)->np.ndarray:

    if dataset_name == 'moons':
        X1, Y1 = make_moons(n_samples=300,random_state=0,noise=noise_rate)
        X2, Y2 = make_moons(n_samples=525, random_state=0, noise=noise_rate)
        X3, Y3 = make_moons(n_samples=675, random_state=0, noise=noise_rate)
        X = np.vstack((X1, X2, X3))
        y1 = np.array([0] * 300)
        y2 = np.array([1] * 525)
        y3 = np.array([2] * 675)        #生成标签数组
        Y = np.hstack((y1, y2, y3))

        # 用kmeans聚成三类，将原始数据集划分成三类。
        y_pred = KMeans(n_clusters=n_clusters, random_state=9).fit_predict(X)

        x = pd.DataFrame(X)
        y_pred = pd.DataFrame(y_pred)
        moons = pd.concat([y_pred,x],axis=1)    #按列拼接datafram/series
        l0 = []
        l1 = []
        l2 = []
        moons.columns = ['0','1','2']       #修改df的列名字
        for index,row in moons.iterrows():              #采样
            if row[0] ==0:
                if len(l0) < 200:
                    l0.append(row)
                    continue
            elif row[0] ==1:
                if len(l1) < 350:
                    l1.append(row)
                    continue
            elif row[0] ==2:
                if len(l2) < 450:
                    l2.append(row)
                    continue

        data = l0+l1+l2
        data = pd.DataFrame(data)       #list to dataframe
        data = data.values         #dataframe to ndarra

        moons = data           #dataframe to ndarray
        X = moons[:,1:]
        Y = moons[:,0]    
        return X,Y
        
    elif dataset_name == 'toy':
        data = pd.read_csv(r'RSDS/DataSet/toy.csv')
        X = data.iloc[:, 1:]
        y = data.iloc[:,0]
    
        return X.values,y.values


def fourclass_data()->np.ndarray:

    data = pd.read_csv(r'fourclass10_change.csv', header=None)  # with an outlier
    # data = pd.read_csv(r'fourclass10.csv', header=None)     # no outlier
    y = data[0]
    X = data[[1, 2]]
    
    X = X.values
    y = y.values
    return X,y


def draw_legend(is_save=False,is_show=False)->None:
    plt.scatter(1,1,label='minority class', c='tan', marker='o', s=25, )
    plt.scatter(1,1,label='majority class', c='darkcyan', marker='o', s=25, )
    plt.scatter(1,1,label='new samples', c='red', s=35, marker='+')
    axes = plt.axes()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.legend(ncol=3,frameon=False)  

    if is_save:  
        plt.savefig(fname='./pdf/'+'lengend'+'.pdf',format='pdf',bbox_inches='tight')        
    if is_show:
        plt.show()


def draw_fourclass(is_save=False,is_show=False)->None:
    X,y = fourclass_data()      #ndarray
    font = {'family': 'Times New Roman',
            'size': 20, }       

    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],c='tan', marker='o', s=10, label='minority: 130')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1],c='darkcyan', marker='o', s=10, label='majority: 430')

    title = 'synthetic data set'
    plt.title(title,font)
    plt.grid()      
    plt.legend()    
    
    if is_save:  
        plt.savefig(fname='./pdf/'+'4_class'+'.pdf',format='pdf',bbox_inches='tight')    
    if is_show:
        plt.show()