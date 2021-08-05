# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 07:10:04 2021

@author: LEGION
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:28:40 2021

@author: -
"""
from library import*
from models import*
def ROC_with_top_feature(data,feature_num,feature_selection_model,classifier):
        
        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models()
#         clff=['MLPClassifier','LinearDiscriminantAnalysis','XGBClassifier','RandomForestClassifier','LogisticRegression','SVM','ExtraTreesClassifier','AdaBoostClassifier','KNeighborsClassifier','GradientBoostingClassifier']
#
        
        if classifier=='all':
            s=0
            for c in range(10):

                #############################
                ## Please edit this part
                #for Xgboost=xgboost, Random Forest=randomforest, Extra tree=extratree
              

                  clf1=clf[c]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                  mean_tpr=[]
                  mean_auc=[]


                  for i in list(range(num_feat)):
                    tl=fsm[0:i+1]
                    
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    for k in range(5):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]   
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        prediction=model.predict_proba(xt1)
                        
                        
                        #prediction = clf1.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
                        fpr, tpr, t = roc_curve(ytest[k], prediction[:, 1])
                        tprs.append(interp(mean_fpr, fpr, tpr))
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)
                        #plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


                    mean_tp = np.mean(tprs, axis=0)
                    mean_au = auc(mean_fpr, mean_tp)
                    mean_tpr.append(mean_tp)
                    mean_auc.append(mean_au)


                  fig1 = plt.figure(figsize=[10,10])

                  for i in range(num_feat):
                    feat_labels = fsm[0:i+1]
                    #every signle ranked feature individually 
                    plt.plot(mean_fpr, mean_tpr[i], marker='.',label=r'Mean ROC with top  %s  feature (AUC = %0.2f )' % (len(feat_labels),mean_auc[i]),linewidth=2.5)


                  plt.title('ROC curves for top features - Classifier:'+clff[s],fontsize = 15)
                  plt.xlabel('False Positive Rate',fontsize = 20)
                  plt.ylabel('True Positive Rate',fontsize = 20)
                  # show the legend
                  plt.legend(fontsize = 11,loc='lower right')
                  # show the plot
                  s=s+1
                  plt.show()
            return
#
        
        else:
              
            

                #############################
                ## Please edit this part
                #for Xgboost=xgboost, Random Forest=randomforest, Extra tree=extratree
              
                  s=classifier
                  clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                  mean_tpr=[]
                  mean_auc=[]


                  for i in list(range(num_feat)):
                    tl=fsm[0:i+1]
                    
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    for k in range(5):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]   
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        prediction=model.predict_proba(xt1)
                        
                        
                        #prediction = clf1.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
                        fpr, tpr, t = roc_curve(ytest[k], prediction[:, 1])
                        tprs.append(interp(mean_fpr, fpr, tpr))
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)
                        #plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


                    mean_tp = np.mean(tprs, axis=0)
                    mean_au = auc(mean_fpr, mean_tp)
                    mean_tpr.append(mean_tp)
                    mean_auc.append(mean_au)


                  fig1 = plt.figure(figsize=[10,10])

                  for i in range(num_feat):
                    feat_labels = fsm[0:i+1]
                    #every signle ranked feature individually 
                    plt.plot(mean_fpr, mean_tpr[i], marker='.',label=r'Mean ROC with top  %s  feature (AUC = %0.2f )' % (len(feat_labels),mean_auc[i]),linewidth=2.5)


                  plt.title('ROC curves for top features - Classifier:'+clff[s],fontsize = 15)
                  plt.xlabel('False Positive Rate',fontsize = 20)
                  plt.ylabel('True Positive Rate',fontsize = 20)
                  # show the legend
                  plt.legend(fontsize = 11,loc='lower right')
                  # show the plot
#                   s=s+1
                  plt.show()
                  return
        
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 22:28:40 2021

@author: -
"""
from library import*
from models import*
def ROC_with_individual_feature(data,feature_num,feature_selection_model,classifier):
        
        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models()
#         clff=['MLPClassifier','LinearDiscriminantAnalysis','XGBClassifier','RandomForestClassifier','LogisticRegression','SVM','ExtraTreesClassifier','AdaBoostClassifier','KNeighborsClassifier','GradientBoostingClassifier']
#
        
        if classifier=='all':
            s=0
            for c in range(10):

                #############################
                ## Please edit this part
                #for Xgboost=xgboost, Random Forest=randomforest, Extra tree=extratree
              

                  clf1=clf[c]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                  mean_tpr=[]
                  mean_auc=[]


                  for i in list(range(num_feat)):
                    tl=fsm[i:i+1]
                    
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    for k in range(5):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]   
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        prediction=model.predict_proba(xt1)
                        
                        
                        #prediction = clf1.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
                        fpr, tpr, t = roc_curve(ytest[k], prediction[:, 1])
                        tprs.append(interp(mean_fpr, fpr, tpr))
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)
                        #plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


                    mean_tp = np.mean(tprs, axis=0)
                    mean_au = auc(mean_fpr, mean_tp)
                    mean_tpr.append(mean_tp)
                    mean_auc.append(mean_au)


                  fig1 = plt.figure(figsize=[10,10])

                  for i in range(num_feat):
                    feat_labels = fsm[i:i+1]
                    #every signle ranked feature individually 
                    plt.plot(mean_fpr, mean_tpr[i], marker='.',label=r'Mean ROC with  %s  feature (AUC = %0.2f )' % (feat_labels,mean_auc[i]),linewidth=2.5)


                  plt.title('ROC curves for top features - Classifier:'+clff[s],fontsize = 15)
                  plt.xlabel('False Positive Rate',fontsize = 20)
                  plt.ylabel('True Positive Rate',fontsize = 20)
                  # show the legend
                  plt.legend(fontsize = 11,loc='lower right')
                  # show the plot
                  s=s+1
                  plt.show()
            return
#
        
        else:
              
            

                #############################
                ## Please edit this part
                #for Xgboost=xgboost, Random Forest=randomforest, Extra tree=extratree
              
                  s=classifier
                  clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                  mean_tpr=[]
                  mean_auc=[]


                  for i in list(range(num_feat)):
                    tl=fsm[i:i+1]
                    
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    for k in range(5):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]   
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        prediction=model.predict_proba(xt1)
                        
                        
                        #prediction = clf1.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
                        fpr, tpr, t = roc_curve(ytest[k], prediction[:, 1])
                        tprs.append(interp(mean_fpr, fpr, tpr))
                        roc_auc = auc(fpr, tpr)
                        aucs.append(roc_auc)
                        #plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))


                    mean_tp = np.mean(tprs, axis=0)
                    mean_au = auc(mean_fpr, mean_tp)
                    mean_tpr.append(mean_tp)
                    mean_auc.append(mean_au)


                  fig1 = plt.figure(figsize=[10,10])

                  for i in range(num_feat):
                    feat_labels = fsm[i:i+1]
                    #every signle ranked feature individually 
                    plt.plot(mean_fpr, mean_tpr[i], marker='.',label=r'Mean ROC with %s  feature (AUC = %0.2f )' % (feat_labels,mean_auc[i]),linewidth=2.5)


                  plt.title('ROC curves for top features - Classifier:'+clff[s],fontsize = 15)
                  plt.xlabel('False Positive Rate',fontsize = 20)
                  plt.ylabel('True Positive Rate',fontsize = 20)
                  # show the legend
                  plt.legend(fontsize = 11,loc='lower right')
                  # show the plot
#                   s=s+1
                  plt.show()
                  return
        