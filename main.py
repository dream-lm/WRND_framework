'''
Author: Zhou Hao
Date: 2022-01-20 20:13:53
LastEditors: ZhouHao 2294776770@qq.com
LastEditTime: 2022-07-12 09:50:22
Description: file content
E-mail: 2294776770@qq.com
'''
from matplotlib import cm
from collections import Counter
# from sklearn.linear_model import PassiveAggressiveClassifier
import nan_zhou
import math
# import NaN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import KDTree
from collections import Counter
from apis import binary_data,fourclass_data,swiss_roll
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import all_smote_v5
from sklearn.preprocessing import scale
import time
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

font = {
        'family':'Times New Roman',
        'size':18,
    }



def main_2d(is_save=False,is_show=False) ->None:
    """
    可视化实验2d图
    """
    
    X, Y = binary_data(data_name='make_moons')  # get dataset
    # X, Y = fourclass_data()
    print(Counter(Y))
    plt.figure(figsize=(15,15),dpi=600)

    Nan = nan_zhou.Natural_Neighbor(X=X,y=Y)
    num,nan_tree =  Nan.algorithm()
    count = Counter(Nan.target)     # 获取少/多数类下标
    c = count.most_common(len(count))
    min_i,maj_i = c[1][0],c[0][0]
    Nan.RelativeDensity(min_i, maj_i)
    relative_cox = np.array(Nan.relative_cox)     # 去噪后的密度权重
    nans = Nan.nan      # 自然邻居
    if num > 5: num -=1
    edges = Nan.nan_edges


    X_max, y_max  = max(X[:,0]),max(X[:,1])
    X_min, y_min  = min(X[:,0]),min(X[:,1])
    X_min, X_max = X_min-(0.1*X_max), X_max+(0.1*X_max)
    y_min, y_max = y_min-(0.1*y_max), y_max+(0.1*y_max)
    colors = [ 'darkcyan' if int(label) == 0 else 'tan'  for label in Y]
    color = ['darkcyan','tan']
    X_more = X[np.where(Y == 1)]
    X_less = X[np.where(Y == 0)]
    tree = KDTree(X_less)   # kdtree of 5nn


    # ***************** draw 5nn *****************
    ax = plt.subplot(221)
    for start, i in enumerate(X):
        if Y[start] == 1: continue  # 只画少数类的边
        neighbors = tree.query([i], k=num+1, return_distance=False)
        neighbors = neighbors[0][1:]      # knn索引
        for neighbor in neighbors:
            x = [X[:,0][start], X[:,0][neighbor]]
            y = [X[:,1][start], X[:,1][neighbor]]
            if Y[start] != Y[neighbor]:
                # red_line=ax.add_line(Line2D(x,y,c='red',linestyle='--',linewidth=0.8,alpha=1))
                pass
            else:
                neighbor_edge=ax.add_line(Line2D(x,y,c=color[int(Y[neighbor])],linestyle='--',linewidth=0.5,alpha=0.5))
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('(a) 5nn',font)
    ax.set_xticklabels([])
    ax.grid()

    min_legend = ax.scatter(X_less[:,0], X_less[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_more[:,0], X_more[:,1],c='tan',marker='*', s=15)
    ax.legend((min_legend,max_legend,neighbor_edge),['minority','majority',"neighbor edge"])
    print("(a) 5NN", num)


    # ***************** draw NaN *****************
    ax = plt.subplot(222)
    for start,end in edges: # start_inde, end_ind            
        if relative_cox[start] > 0 :
            x = [X[:,0][start], X[:,0][end]]
            y = [X[:,1][start], X[:,1][end]]
            if Y[start] != Y[end]:  # 异类边
                red_line = ax.add_line(Line2D(x,y,c='red',linestyle='--',linewidth=0.8,alpha=1))
            else:
                NaN_line = ax.add_line(Line2D(x,y,c=color[int(Y[end])],linestyle='--',linewidth=0.5,alpha=0.5))
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticklabels([])  # 隐藏坐标轴刻度，同时也可以呃画网格
    ax.set_yticklabels([])
    ax.set_title('(b) NaN',font)
    ax.scatter(X[:,0],X[:,1],c=colors,marker='o', s=5)
    ax.grid()

    min_legend = ax.scatter(X_less[:,0], X_less[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_more[:,0], X_more[:,1],c='tan',marker='*', s=15)
    ax.legend((min_legend,max_legend,red_line,NaN_line),['minority','majority','Heterogeneous edge',"Homogeneous edge"])
    print("(b) NaN", num)


    # ***************** draw smote *****************
    ax = plt.subplot(223)
    for start, i in enumerate(X):
        if Y[start] == 1: continue  # 只画少数类的边
        neighbors = tree.query([i], k=num+1, return_distance=False)
        neighbors = neighbors[0][1:]      # knn索引
        for neighbor in neighbors:
            x = [X[:,0][start], X[:,0][neighbor]]
            y = [X[:,1][start], X[:,1][neighbor]]
            if Y[start] != Y[neighbor]:
                pass
            else:
                neighbor_edge = ax.add_line(Line2D(x,y,c=color[int(Y[neighbor])],linestyle='--',linewidth=0.5,alpha=0.5))
    
    model = SMOTE(random_state=42)
    X_res, y_res = model.fit_resample(X, Y)
    clf = SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(X_res,y_res)

    xx, yy = np.meshgrid(np.linspace(X_min, X_max, 1000),np.linspace(y_min, y_max, 1000))
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title('(c) SMOTE',font)

    ax.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
            origin='lower', cmap='PuBu')
    ax.contourf(xx, yy, Z,alpha=0.5,cmap="PuBu")
    ax.contour(xx, yy, Z,linewidths=0.5,cmap='winter')

    min_legend = ax.scatter(X_less[:,0], X_less[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_more[:,0], X_more[:,1],c='tan',marker='*', s=15)
    
    X_new = pd.DataFrame(X_res).iloc[len(X):, :]
    ax.grid()
    new_legend = ax.scatter(X_new[0], X_new[1], c='black', s=5, marker='+')
    ax.legend((min_legend,max_legend,new_legend,neighbor_edge),['minority','majority','new samples',"neighbor edge"])
    print('(c) SMOTE')


    # ***************** draw Nan Oversample *****************
    ax = plt.subplot(224)
    for start,end in edges: # start_inde, end_ind            
        if relative_cox[start] > 0:
            x = [X[:,0][start], X[:,0][end]]
            y = [X[:,1][start], X[:,1][end]]
            if Y[start] != Y[end]:
                red_line = ax.add_line(Line2D(x,y,c='red',linestyle='--',linewidth=0.8,alpha=1))
            else:
                NaN_line = ax.add_line(Line2D(x,y,c=color[int(Y[end])],linestyle='--',linewidth=0.5,alpha=0.5))

    model = all_smote_v5.SMOTE(random_state=42,nans=nans,weight=relative_cox )
    X_res, y_res = model.fit_resample(X, Y)  
    clf = SVC(C=1, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(X_res,y_res)

    xx, yy = np.meshgrid(np.linspace(X_min, X_max, 1000),np.linspace(y_min, y_max, 1000))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_yticklabels([])
    ax.set_title('(d) SMOTE-WNND',font)

    ax.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
            origin='lower', cmap='PuBu')
    ax.contourf(xx, yy, Z,alpha=0.5,cmap = 'PuBu')
    ax.contour(xx,yy,Z,linewidths=0.5,cmap='winter') 

    min_legend = ax.scatter(X_less[:,0], X_less[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_more[:,0], X_more[:,1],c='tan',marker='*', s=15)
    X_new = pd.DataFrame(X_res).iloc[len(X):, :]
    ax.grid()
    new_legend = ax.scatter(X_new[0], X_new[1], c='black', s=5, marker='+')
    ax.legend((min_legend,max_legend,new_legend,red_line,NaN_line),['minority','majority','new samples','Heterogeneous edge',"Homogeneous edge"])
    print('(d) Nan SMOTE')


    plt.tight_layout()
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        plt.savefig(fname=r"PDF/2d/"+ now + str(num) + '.pdf',format='pdf',bbox_inches='tight')
    if is_show:
        plt.show()



def main_3d(is_save=False,is_show=False) ->None:
    """可视化实验3d图"""
    # get data
    X,Y = swiss_roll()
    X = scale(X=X,with_mean=True,with_std=True,copy=True)   # 标准化
    print(Counter(Y))


    # get num,kdtree,edges
    Nan = nan_zhou.Natural_Neighbor(X=X,y=Y)
    num,nan_tree =  Nan.algorithm()
    if num > 5: num -=1
    edges = Nan.nan_edges
    kd_tree = KDTree(X[np.where(Y == 0.0)])


    # get rd and nans
    count = Counter(Nan.target)     # 获取少/多数类下标
    c = count.most_common(len(count))
    min_i,maj_i = c[1][0],c[0][0]
    Nan.RelativeDensity(min_i, maj_i)
    relative_cox = np.array(Nan.relative_cox)      # 去噪后的密度权重
    nans = Nan.nan      # 自然邻居
    relative_cox[np.isinf(relative_cox)] = 0

    # set params to draw
    X_max, y_max = max(X[:,0]),max(X[:,1])
    X_min, y_min = min(X[:,0]),min(X[:,1])
    Z_min, Z_max = min(X[:,2]),max(X[:,2])
    X_min, X_max = X_min-0.1, X_max+0.1
    y_min, y_max = y_min-0.1, y_max+0.1
    Z_min, Z_max = Z_min-0.1, Z_max+0.1
    colors = [ 'darkcyan' if int(label) == 0 else 'tan'  for label in Y]
    color = ['darkcyan','tan']

    # get figure
    plt.figure(figsize=(20,15),dpi=800)


    # draw knn ***********************************************************
    ax = plt.subplot(221, projection='3d')
    for start,i in enumerate(X):
        if Y[start] != 0: continue
        neighbors = kd_tree.query([i], k=num+1, return_distance=False)
        neighbors = neighbors[0][1:]      # knn索引
        for neighbor in neighbors:
            x = np.array([X[:,0][start], X[:,0][neighbor]])
            y = np.array([X[:,1][start], X[:,1][neighbor]])
            z = np.array([X[:,2][start], X[:,2][neighbor]])
            if Y[start] != Y[neighbor]:
                # ax.plot(x, y, z, c='red',linestyle='--',linewidth=0.8,alpha=1)
                continue
            else:
                ax.plot(x, y, z, c=color[int(Y[neighbor])],linestyle='--',linewidth=0.8,alpha=1)
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(Z_min, Z_max)
    ax.set_xlabel('X',font)
    ax.set_ylabel('Y',font)
    ax.set_zlabel('Z',font)
    ax.set_title('(a) 5nn',font)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, marker='o',s=15)
    ax.view_init(elev=15,azim=-76)  # 调整可视化角度
    print("(a) 5nn", num)


    #  draw Nan ***********************************************************
    ax = plt.subplot(222,projection='3d')
    for start,end in edges:
        if Y[start] != 0: continue
        x = [X[:,0][start], X[:,0][end]]
        y = [X[:,1][start], X[:,1][end]]
        z = [X[:,2][start], X[:,2][end]]
        if Y[start] != Y[end]:
            ax.plot(x, y, z, c='red',linestyle='--',linewidth=0.8,alpha=1)
        else:
            ax.plot(x, y, z, c=color[int(Y[end])],linestyle='--',linewidth=0.8,alpha=1)
    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(Z_min, Z_max)
    ax.set_xlabel('X',font)
    ax.set_ylabel('Y',font)
    ax.set_zlabel('Z',font)
    ax.set_title('(b) NaN',font)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, marker='o',s=15)
    ax.view_init(elev=15,azim=-76)
    print("(b) NaN:", num)



    # draw smote ***********************************************************
    ax = plt.subplot(223,projection='3d')
    model = SMOTE(random_state=42)
    X_res, y_res = model.fit_resample(X, Y)

    clf = SVC(C=1,random_state=42, probability=True,kernel='linear',gamma=20,decision_function_shape='ovr')  # kernel='linear', gamma='auto',
    clf.fit(X_res,y_res)

    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.linspace(X_min, X_max,1000),np.linspace(y_min, y_max, 1000))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  #计算超平面

    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(Z_min, Z_max)
    ax.set_xlabel('X',font)
    ax.set_ylabel('Y',font)
    ax.set_zlabel('Z',font)
    ax.set_title('(c) SMOTE',font)
    ax.scatter(X[:,0],X[:,1],X[:, 2],c=colors,marker='o', s=15)
    X_new = pd.DataFrame(X_res).iloc[len(X):, :]
    ax.scatter(X_new[0], X_new[1],X_new[2], c='red', s=15, marker='+')
    ax.plot_surface(xx,yy,Z1,alpha=0.6)
    ax.view_init(elev=11,azim=-84)
    print('(c) SMOTE')

    
    # draw Nan Oversample ***********************************************************
    ax = plt.subplot(224,projection='3d')
    model = all_smote_v5.SMOTE(random_state=42,nans=nans,weight=relative_cox )
    X_res, y_res = model.fit_resample(X, Y)
    
    clf = SVC(random_state=42, probability=True,kernel='linear')  
    clf.fit(X_res,y_res)

    b=clf.intercept_    # 超平面常数
    w=clf.coef_         # 超平面权重系数
    xx, yy = np.meshgrid(np.arange(X_min,X_max,0.02),  np.arange(y_min,y_max,0.02))
    Z1= -w[0,0]/w[0,2]*xx-w[0,1]/w[0,2]*yy-b[0]/w[0,2]  #计算超平面

    ax.set_xlim(X_min, X_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(Z_min, Z_max)
    ax.set_xlabel('X',font)
    ax.set_ylabel('Y',font)
    ax.set_zlabel('Z',font)
    ax.set_title('(d) SMOTE-WNND',font)
    ax.scatter(X[:,0],X[:,1],X[:, 2],c=colors,marker='o', s=15)
    X_new = pd.DataFrame(X_res).iloc[len(X):, :]
    ax.scatter(X_new[0], X_new[1],X_new[2], c='red', s=15, marker='+')
    ax.plot_surface(xx,yy,Z1,alpha=0.6)
    ax.view_init(elev=11,azim=-84)
    print('(d) WNND+SMOTE')


    # Save and show
    plt.tight_layout()
    plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        plt.savefig(fname=r"PDF/3d/"+ now +str(num) + '.pdf',format='pdf',bbox_inches='tight')
    if is_show:
        plt.show()


def main_motivation(is_save=0,is_show=1):
    # motivation中不需要采样

    X, Y = binary_data(data_name='make_moons')  # get dataset
    print(Counter(Y))
    plt.figure(figsize=(10,10),dpi=600)

    Nan = nan_zhou.Natural_Neighbor(X=X,y=Y)
    num,_ =  Nan.algorithm()
    count = Counter(Nan.target)     # 获取少/多数类下标
    c = count.most_common(len(count))
    min_i,maj_i = c[1][0],c[0][0]
    Nan.RelativeDensity(min_i, maj_i)
    relative_cox = np.array(Nan.relative_cox)     # 去噪后的密度权重
    if num > 5: num -=1
    edges = Nan.nan_edges
    color = ['darkcyan','tan']
    X_min = X[np.where(Y == 0)]
    X_maj = X[np.where(Y == 1)]
    tree = KDTree(X_min)



    # draw 3nn
    ax = plt.subplot(221)  
    model = SMOTE(random_state=42, k_neighbors=3 )
    X_res, y_res = model.fit_resample(X, Y)
    
    for start,i in enumerate(X_min):
        neighbors = tree.query([i], k=3+1, return_distance=False)
        neighbors = neighbors[0][1:]      # knn索引
        for neighbor in neighbors:
            x = [X_min[:,0][start], X_min[:,0][neighbor]]
            y = [X_min[:,1][start], X_min[:,1][neighbor]]
            line_legend = ax.add_line(Line2D(x,y,c='darkcyan',linestyle='--',linewidth=0.8,alpha=0.5))

    ax.set_xlim(0.5, 2)
    ax.set_ylim(-0.5, 1)
    ax.set_title('(a) 3NN',font)
    
    min_legend = ax.scatter(X_min[:,0], X_min[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_maj[:,0], X_maj[:,1],c='tan',marker='*', s=15)
    X_new = pd.DataFrame(X_res).iloc[len(X):, :]
    new_legend = ax.scatter(X_new[0], X_new[1], c='red', s=15, marker='+')
    ax.grid()
    ax.legend((min_legend,max_legend,new_legend,line_legend),['minority','majority','new samples',r"neighbor edge"])
    print("(a) 3NN")


    # draw 5nn
    ax = plt.subplot(222)
    model = SMOTE(random_state=42, k_neighbors=5 )
    X_res, y_res = model.fit_resample(X, Y)

    for start,i in enumerate(X_min):
        neighbors = tree.query([i], k=5+1, return_distance=False)
        neighbors = neighbors[0][1:]      # knn索引
        for neighbor in neighbors:
            x = [X_min[:,0][start], X_min[:,0][neighbor]]
            y = [X_min[:,1][start], X_min[:,1][neighbor]]
            ax.add_line(Line2D(x,y,c='darkcyan',linestyle='--',linewidth=0.8,alpha=0.5))

    ax.set_xlim(0.5, 2)
    ax.set_ylim(-0.5, 1)
    ax.set_title('(b) 5NN',font)

    min_legend = ax.scatter(X_min[:,0], X_min[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_maj[:,0], X_maj[:,1],c='tan',marker='*', s=15)
    X_new = pd.DataFrame(X_res).iloc[len(X):, :]
    new_legend = ax.scatter(X_new[0], X_new[1], c='red', s=15, marker='+')
    ax.grid()
    ax.legend((min_legend,max_legend,new_legend,line_legend),['minority','majority','new samples',r"neighbor edge"])
    print("(b) 5nn")



    # draw NaN ----------------------------------------
    ax = plt.subplot(223)
    for start,end in edges: # start_inde, end_ind            
        if Y[start] == Y[end] == 1:
            continue
        x = [X[:,0][start], X[:,0][end]]
        y = [X[:,1][start], X[:,1][end]]
        if Y[start] != Y[end]:
            red_line = ax.add_line(Line2D(x,y,c='Red',linestyle='--',linewidth=0.8,alpha=0.5))
        else:
            NaN_line = ax.add_line(Line2D(x,y,c=color[int(Y[end])],linestyle='--',linewidth=0.8,alpha=0.5))

    ax.set_xlim(0.5, 2)
    ax.set_ylim(-0.5, 1)
    ax.set_title('(c) NaN',font)

    min_legend = ax.scatter(X_min[:,0], X_min[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_maj[:,0], X_maj[:,1],c='tan',marker='*', s=15)
    ax.grid()
    ax.legend((min_legend,max_legend,NaN_line,red_line),
        ['minority','majority','NaN edge','heterogeneous edge'],
        fontsize=8)
    print("(c) NaN")


    # 画密度圆
    ax = plt.subplot(224)
    max_cox = max(relative_cox)
    min_cox = min(relative_cox[np.where(relative_cox > 0)])
    for i in range(len(X)):
        if Y[i] == 0 and relative_cox[i] > 0:
            circle = plt.Circle((X[i][0],X[i][1]), 0.06, color = 'darkcyan', alpha = (relative_cox[i] - min_cox)/(max_cox - min_cox))#圆心，半径，颜色，α
            circle_side = plt.Circle((X[i][0],X[i][1]), 0.06, color = 'black',alpha=0.2,fill=False)#圆心，半径，颜色，α
            ax.add_patch(circle)
            ax.add_patch(circle_side)
    
    ax.set_xlim(0.5, 2)
    ax.set_ylim(-0.5, 1)
    ax.set_title('(d) Density of NaN',font)
    min_legend = ax.scatter(X_min[:,0], X_min[:,1],c='darkcyan',marker='^', s=15)
    max_legend = ax.scatter(X_maj[:,0], X_maj[:,1],c='tan',marker='*', s=15)
    ax.grid()
    ax.legend((min_legend,max_legend),['minority','majority'])
    print("(d) Density")


    plt.tight_layout()
    if is_save:
        now = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
        plt.savefig(fname="PDF/motivation/"  + now +str(num) + '.pdf',format='pdf',bbox_inches='tight')
    if is_show:
        plt.show()


if __name__ == '__main__':
    main_2d(is_save=1,is_show=0)
    # main_3d(is_save=1,is_show=0)
    # main_motivation(is_save=1,is_show=0)