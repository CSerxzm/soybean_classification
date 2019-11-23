# -*- coding: utf-8 -*-

import pandas as pd
import numpy
from sklearn.preprocessing import Imputer


def load_data_set_train():
    url = "../data/soybean-large.data"
    names = ['Class', 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
             'severity', 'seed-tmt', 'germination', 'plant-growth',
             'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf',
             'leaf-mild', 'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies',
             'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots',
             'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
    dataset = pd.read_csv(url, names=names)
    dataset = dataset.replace({'?':numpy.nan}) 
    df1=dataset.iloc[:, 1:]
    df2=dataset.iloc[:, :1]
    imr= Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0 )
    imr = imr.fit(df1)
    imputed_data = imr.transform(df1.values)
    df = pd.DataFrame(imputed_data) 
    df=pd.concat([df2,df],axis=1)
    return df

def load_data_set_test():
    url = "../data/soybean-large.data"
    names = ['Class', 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged',
             'severity', 'seed-tmt', 'germination', 'plant-growth',
             'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size', 'leaf-shread', 'leaf-malf',
             'leaf-mild', 'stem', 'lodging', 'stem-cankers', 'canker-lesion', 'fruiting-bodies',
             'external decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods', 'fruit spots',
             'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots']
    df = pd.read_csv(url, names=names)
    return df