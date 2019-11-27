# -*- coding: utf-8 -*-
import datahandler

from sklearn import model_selection
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt 
from  mpl_toolkits import mplot3d

#讨论随机数森林影响
def discussion_rfc(x_start,x_end,y_start,y_end):
    
    dataset_train = load_data_set_train()
    array = dataset_train.values
    x= array[:,1:len(list(dataset_train))-1]
    y= array[:,0]
    validation_size = 0.2
    seed = 7
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,random_state=seed)
    
    all_mae_histories = [[i*j for j in range(x_start,x_end)] for i in range(y_start,y_end)]
    i=0
    for tree_number in range(y_start,y_end):
        j=0
        for depth in range(x_start,x_end):
            rf = RandomForestClassifier(n_estimators=tree_number, max_depth=depth)
            # 训练模式
            rf.fit(x_train, y_train)
            predictions = rf.predict(x_validation)
            accuracy_aux = accuracy_score(y_validation, predictions)
            all_mae_histories[i][j]=accuracy_aux
            j=j+1
        i=i+1
    show_3d(x_start,x_end,y_start,y_end,all_mae_histories) 
    
def show_plt(all_mae_histories,x_start,x_end):
    x=range(x_start,x_end)
    plt.figure()  
    plt.plot(x,all_mae_histories)
    
def show_3d(x_start,x_end,y_start,y_end,all_mae_histories):
    fig = plt.figure()  #定义新的三维坐标轴
    ax3 = plt.axes( projection='3d')

    xx = np.arange(x_start,x_end)
    yy = np.arange(y_start,y_end)
    X,Y = np.meshgrid(xx, yy)
    Z=np.array(all_mae_histories) 
    #作图
    ax3.plot_surface(X,Y,Z,cmap='rainbow')
    plt.show()


discussion_rfc(5,50,5,50)    
#show_3d(1,1,1,1,1)       
    
