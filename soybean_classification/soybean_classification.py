from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
import os
from sklearn.externals import joblib

from sklearn import model_selection
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)

import datahandler

def all_Algorithms():
    all_algorithms_score=[]
    all_algorithms_score_avg=[]

    all_algorithms_name=["DecisionTree","MLP","RandomForest","NaiveBayes","SVM"]
    
    #load data
    dataset_train = load_data_set_train()
    array = dataset_train.values
    x= array[:,1:len(list(dataset_train))-1]
    y= array[:,0]
    validation_size = 0.2
    seed = 8
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,random_state=seed)

    accuracy,average_mae_history=DecisionTree_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)
    
    accuracy,average_mae_history=MLPClassifier_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)
    
    accuracy,average_mae_history=RandomForestClassifier_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)
    
    accuracy,average_mae_history=NaiveBayes_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)
    
    accuracy,average_mae_history=SVM_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    all_algorithms_score_avg.append(average_mae_history)
    
    return all_algorithms_name,all_algorithms_score,all_algorithms_score_avg,list(dataset_train),list(dataset_train.mean())

def DecisionTree_Algorithms(x_train, x_validation, y_train, y_validation):   
    #DecisionTree Algorithms
    all_mae_histories = []
    k = 10
    num_val_samples = len(x_train) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y_train[:i * num_val_samples],
             y_train[(i + 1) * num_val_samples:]],
            axis=0)
        # 构建 Keras 模型
        dtc = DecisionTreeClassifier()
        dtc.fit(partial_train_data, partial_train_targets)
        predictions = dtc.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = dtc.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    save_model(dtc,"DecisionTree")
    print("DecisionTree:",accuracy)
    
    return accuracy,average_mae_history
    
def MLPClassifier_Algorithms(x_train, x_validation, y_train, y_validation):    
    #MLPClassifier Algorithms
    seed = 7
    all_mae_histories = []
    k = 10
    num_val_samples = len(x_train) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y_train[:i * num_val_samples],
             y_train[(i + 1) * num_val_samples:]],
            axis=0)
        mlp = MLPClassifier(random_state=seed, solver='lbfgs')
        mlp.fit(partial_train_data, partial_train_targets)
        predictions = mlp.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = mlp.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    save_model(mlp,"MLPClassifier")
    print("MLPClassifier:",accuracy)
    return accuracy,average_mae_history

def RandomForestClassifier_Algorithms(x_train, x_validation, y_train, y_validation):         
    #SVM Algorithms
    all_mae_histories = []
    k = 10
    num_val_samples = len(x_train) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y_train[:i * num_val_samples],
             y_train[(i + 1) * num_val_samples:]],
            axis=0)

        rf = RandomForestClassifier(n_estimators=10, max_depth=10)
        rf.fit(partial_train_data, partial_train_targets)
        predictions = rf.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均,没有使用

    average_mae_history = np.mean(all_mae_histories)

    predictions = rf.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    save_model(rf,"RandomForestClassifier")
    print("RandomForestClassifier:",accuracy)
    return accuracy,average_mae_history


def NaiveBayes_Algorithms(x_train, x_validation, y_train, y_validation):     
    #Naive Bayes Algorithms
    all_mae_histories = []
    k = 10
    num_val_samples = len(x_train) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y_train[:i * num_val_samples],
             y_train[(i + 1) * num_val_samples:]],
            axis=0)
        # 构建 Keras 模型
        nb = GaussianNB()
        # 训练模式
        nb.fit(partial_train_data, partial_train_targets)
        predictions = nb.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = nb.predict(x_validation)
    accuracy= accuracy_score(y_validation, predictions)
    save_model(nb,"NaiveBayes")
    print("Naive Bayes:",accuracy)
    return accuracy,average_mae_history

def SVM_Algorithms(x_train, x_validation, y_train, y_validation):         
    #SVM Algorithms
    all_mae_histories = []
    k = 10
    num_val_samples = len(x_train) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [x_train[:i * num_val_samples],
             x_train[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [y_train[:i * num_val_samples],
             y_train[(i + 1) * num_val_samples:]],
            axis=0)
        # 构建 Keras 模型
        svc = SVC(kernel='poly', gamma="auto")
        # 训练模式
        svc.fit(partial_train_data, partial_train_targets)
        predictions = svc.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均,没有使用
    show_plt(all_mae_histories,"SVM")
    average_mae_history = np.mean(all_mae_histories)

    predictions = svc.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    save_model(svc,"SVM")
    print("SVM:",accuracy)
    return accuracy,average_mae_history

def K_vertify_knn(train_data,train_targets,knnnumber):
    #K折验证，适用于数据集较少的数据集
    all_mae_histories = []
    k = 10
    num_val_samples = len(train_data) // k
    for i in range(k):
        # 准备验证数据，第K个分区的数据
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        # 构建 Keras 模型
        knn = KNeighborsClassifier(n_neighbors=knnnumber)
        # 训练模式
        knn.fit(partial_train_data, partial_train_targets)
        predictions = knn.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均
    average_mae_history = np.mean(all_mae_histories)
    return average_mae_history

def save_model(model_temp,model_name):
    dirs = "../testModel"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    joblib.dump(model_temp, dirs+"/"+model_name)
    
def show_plt(all_mae_histories,model_name):
    
    plt.figure()
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    X=range(1,11)
    plt.bar(X,all_mae_histories,width=0.3,label="vaildate_score")
    for x,y in zip(X,all_mae_histories): 
        plt.text(x,y+0.01,'%.3f' %y, ha='center',va='bottom')
    plt.xlabel(u'K折次数',FontProperties=font)
    plt.ylabel(u'精确度',FontProperties=font)
    plt.title(u'K折精确度变化('+model_name+')', FontProperties=font)
    plt.show()
