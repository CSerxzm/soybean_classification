from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pickle
import datahandler


def all_Algorithms():
    all_algorithms_score=[]
    all_algorithms_name=["KNN","DecisionTree","MLPClassifier","Naive Bayes","SVM"]

    #load data
    dataset_train = load_data_set_train()
    array = dataset_train.values
    x = array[:, 1:35]
    y = array[:, 0]
    validation_size = 0.20
    seed = 7
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,random_state=seed)
    accuracy=knn_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    
    accuracy=DecisionTree_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    
    accuracy=MLPClassifier_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    
    accuracy=NaiveBayes_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    
    accuracy=SVM_Algorithms(x_train, x_validation, y_train, y_validation)
    all_algorithms_score.append(accuracy)
    
    return all_algorithms_name,all_algorithms_score


def knn_Algorithms(x_train, x_validation, y_train, y_validation):                                                                                       
    # knn Algorithms
    best_k,max_value= choose_best_k_to_knn(x_train, y_train, x_validation, y_validation)
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)
    predictions = knn.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    print("KNN:",accuracy)
    return accuracy

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
        # 训练模式
        dtc.fit(partial_train_data, partial_train_targets)
        predictions = dtc.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = dtc.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    print("\nDecisionTree:",accuracy)
    return accuracy
    
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
        # 构建 Keras 模型
        mlp = MLPClassifier(random_state=seed, solver='lbfgs')
        # 训练模式
        mlp.fit(partial_train_data, partial_train_targets)
        predictions = mlp.predict(val_data)
        accuracy_aux = accuracy_score(val_targets, predictions)
        all_mae_histories.append(accuracy_aux)
    #K折验证分数平均,没有使用
    average_mae_history = np.mean(all_mae_histories)

    predictions = mlp.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    print("\nMLPClassifier:",accuracy)
    return accuracy
     
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
    print("\nNaive Bayes:",accuracy)
    return accuracy

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
    average_mae_history = np.mean(all_mae_histories)

    predictions = svc.predict(x_validation)
    accuracy = accuracy_score(y_validation, predictions)
    print("\nSVM Bayes:",accuracy)
    return accuracy

def choose_best_k_to_knn(x_train, y_train, x_validation, y_validation):
    all_mae_histories = []
    accuracy = 0
    k = 1
    for i in range(1, 30):
        average_mae_history=K_vertify_knn(x_train,y_train,i)
        all_mae_histories.append(average_mae_history)
    index,max_value = smooth_curve(all_mae_histories)
    return index,max_value

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

#绘制验证分数，找到最佳k
def smooth_curve(points, factor=0.9):
    index=-1
    max_value=-999
    plt.plot(range(1, len(points) + 1), points)
    plt.xlabel('k value')
    plt.ylabel('grade')
    plt.show()
    for i, val in enumerate(points):
        if max_value < val:
            index=i
            max_value=val
    return index,max_value