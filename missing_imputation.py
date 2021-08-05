# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 23:31:13 2021

@author: -Tawsif
"""
from library import*
import numpy as np

def missing_imputaion(x,imputer='none'):
    xt=x
    if imputer=='knn':
        
        X = xt
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        Knn_data=imputer.fit_transform(X)
        X1=pd.DataFrame(Knn_data)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    elif imputer=='mice':
        Mice_data=MICE().fit_transform(xt)
        X1=pd.DataFrame(Mice_data)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    elif imputer=='randomforest':
        imputer = MissForest()
        Rf = imputer.fit_transform(xt)
        X1=pd.DataFrame(Rf)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    else:
        X1=xt.dropna(axis=0)
        return(X1)

    

    