# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def showbar(all_algorithms_name,all_algorithms_score,all_algorithms_score_avg):
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    x = np.arange(5)
    plt.bar(x-0.15,all_algorithms_score,width=0.3,label="vaildate_score")
    plt.bar(x+0.15,all_algorithms_score_avg,width=0.3,label="avg_score")
    plt.xticks(np.arange(5),all_algorithms_name, rotation=90)#rotation控制倾斜角度
    plt.xlabel(u'algorithms',FontProperties=font)
    plt.ylabel(u'score',FontProperties=font)
    plt.title(u'算法结果', FontProperties=font)
    plt.show()
def get_maxalgorithms(all_algorithms_name,all_algorithms_score):
    max_index=-1
    max_value=-999
    for i in all_algorithms_score:
        if max_value < i :
            max_index=all_algorithms_score.index(i)
            max_value=i
    return max_value,all_algorithms_name[max_index]                  

if __name__ == '__main__':
    
    all_algorithms_name,all_algorithms_score,all_algorithms_score_avg=all_Algorithms()
    
    showbar(all_algorithms_name,all_algorithms_score,all_algorithms_score_avg)
    accuracy,algorithm_chosen=get_maxalgorithms(all_algorithms_name,all_algorithms_score)
    
    print("\nAlgorithm Chosen: " + algorithm_chosen)
    print("Accuracy: %f" % accuracy)
    
    print("\n测试部分:")
    dataset_test = load_data_set_test()