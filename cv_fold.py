# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 07:18:45 2021

@author: Tawsif
"""

def cv_fold(X1,yt,n_splits=5,shuffle=False):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    smote=SMOTE()
    cc=X1.columns
    X=np.array(X1)
    y=np.array(yt)
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=shuffle)
    skf.get_n_splits(X, y)
    #StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    for train_index, test_index in skf.split(X, y):
#          print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]
            x1,y1= smote.fit_resample(X_train, y_train)
            xtrain.append(x1)
            xtest.append(X_test)
            ytrain.append(y1)
            ytest.append(y_test)
    d={'data':(xtrain,xtest,ytrain,ytest),'index':cc}
    return d