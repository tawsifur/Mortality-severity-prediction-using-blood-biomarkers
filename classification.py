# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 06:28:35 2021

@author: Tawsif
"""

from library import*
from models import*

def classification_with_top_feature(data,feature_num,feature_selection_model,classifier):
        
        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models()
        
        
        if classifier=='all':
            l=0
            for c in range(10):

                clf1=clf[c]  
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]

                feat=[]
                for i in list(range(num_feat)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]
               
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
                        y_pr=model.predict(xt1)
                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    ac = accuracy_score(y2, y_pred)
                    #y_pred = cross_val_predict(clf, X, y, cv=5)
                    pr = precision_score(y2, y_pred, average='weighted')
                    rc = recall_score(y2, y_pred, average='weighted')
                    sp = recall_score(y2, y_pred, average='weighted',pos_label=0)
                    f1 = f1_score(y2, y_pred, average='weighted')
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'



                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)
                    acc=1.96*sqrt((ac*(1-ac))/len(y2))
                    prc=1.96*sqrt((pr*(1-pr))/len(y2))
                    rcl=1.96*sqrt((rc*(1-rc))/len(y2))
                    f11=1.96*sqrt((f1*(1-f1))/len(y2))
                    spc=1.96*sqrt((sp*(1-sp))/len(y2))

#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#                     #print(classification_report(y, y_pred, digits=2))
#                     print("accuracy: %0.2f (+/- %0.2f)" % (ac.mean(), acc))
#                     print("precision: %0.2f (+/- %0.2f)" % (pr.mean(), prc))
#                     print("recall: %0.2f (+/- %0.2f)" % (rc.mean(), rcl))
#                     print("f1: %0.2f (+/- %0.2f)" % (f1.mean(), f11))
#                     print("specificity: %0.2f (+/- %0.2f)" % (sp.mean(), spc))


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
                l=l+1
                print('---------------------------------------------------------------------')
                         
            return
        else:
                 
                clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                l=classifier
#             l=0
            

#                 clf1=clf[c]  
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]

                feat=[]
                for i in list(range(num_feat)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]
               
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
                        y_pr=model.predict(xt1)
                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    ac = accuracy_score(y2, y_pred)
                    #y_pred = cross_val_predict(clf, X, y, cv=5)
                    pr = precision_score(y2, y_pred, average='weighted')
                    rc = recall_score(y2, y_pred, average='weighted')
                    sp = recall_score(y2, y_pred, average='weighted',pos_label=0)
                    f1 = f1_score(y2, y_pred, average='weighted')
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'



                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)
                    acc=1.96*sqrt((ac*(1-ac))/len(y2))
                    prc=1.96*sqrt((pr*(1-pr))/len(y2))
                    rcl=1.96*sqrt((rc*(1-rc))/len(y2))
                    f11=1.96*sqrt((f1*(1-f1))/len(y2))
                    spc=1.96*sqrt((sp*(1-sp))/len(y2))

#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#                     #print(classification_report(y, y_pred, digits=2))
#                     print("accuracy: %0.2f (+/- %0.2f)" % (ac.mean(), acc))
#                     print("precision: %0.2f (+/- %0.2f)" % (pr.mean(), prc))
#                     print("recall: %0.2f (+/- %0.2f)" % (rc.mean(), rcl))
#                     print("f1: %0.2f (+/- %0.2f)" % (f1.mean(), f11))
#                     print("specificity: %0.2f (+/- %0.2f)" % (sp.mean(), spc))


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
#                 l=l+1
                print('---------------------------------------------------------------------')
                         
                return
                    
                    

from library import*
from models import*

def classification_with_individual_feature(data,feature_num,feature_selection_model,classifier):
        
        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models()
        
        
        if classifier=='all':
            l=0
            for c in range(10):

                clf1=clf[c]  
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]

                feat=[]
                for i in list(range(num_feat)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[i:i+1]
               
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
                        y_pr=model.predict(xt1)
                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    ac = accuracy_score(y2, y_pred)
                    #y_pred = cross_val_predict(clf, X, y, cv=5)
                    pr = precision_score(y2, y_pred, average='weighted')
                    rc = recall_score(y2, y_pred, average='weighted')
                    sp = recall_score(y2, y_pred, average='weighted',pos_label=0)
                    f1 = f1_score(y2, y_pred, average='weighted')
                    t=fsm[i:i+1][0]
                    feature_no= t



                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)
                    acc=1.96*sqrt((ac*(1-ac))/len(y2))
                    prc=1.96*sqrt((pr*(1-pr))/len(y2))
                    rcl=1.96*sqrt((rc*(1-rc))/len(y2))
                    f11=1.96*sqrt((f1*(1-f1))/len(y2))
                    spc=1.96*sqrt((sp*(1-sp))/len(y2))

#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#                     #print(classification_report(y, y_pred, digits=2))
#                     print("accuracy: %0.2f (+/- %0.2f)" % (ac.mean(), acc))
#                     print("precision: %0.2f (+/- %0.2f)" % (pr.mean(), prc))
#                     print("recall: %0.2f (+/- %0.2f)" % (rc.mean(), rcl))
#                     print("f1: %0.2f (+/- %0.2f)" % (f1.mean(), f11))
#                     print("specificity: %0.2f (+/- %0.2f)" % (sp.mean(), spc))


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
                l=l+1
                print('---------------------------------------------------------------------')
                         
            return
        else:
                 
                clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                l=classifier
#             l=0
            

#                 clf1=clf[c]  
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]

                feat=[]
                for i in list(range(num_feat)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[i:i+1]
               
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
                        y_pr=model.predict(xt1)
                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    ac = accuracy_score(y2, y_pred)
                    #y_pred = cross_val_predict(clf, X, y, cv=5)
                    pr = precision_score(y2, y_pred, average='weighted')
                    rc = recall_score(y2, y_pred, average='weighted')
                    sp = recall_score(y2, y_pred, average='weighted',pos_label=0)
                    f1 = f1_score(y2, y_pred, average='weighted')
                    t=fsm[i:i+1][0]
                    feature_no= t



                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)
                    acc=1.96*sqrt((ac*(1-ac))/len(y2))
                    prc=1.96*sqrt((pr*(1-pr))/len(y2))
                    rcl=1.96*sqrt((rc*(1-rc))/len(y2))
                    f11=1.96*sqrt((f1*(1-f1))/len(y2))
                    spc=1.96*sqrt((sp*(1-sp))/len(y2))

#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#                     #print(classification_report(y, y_pred, digits=2))
#                     print("accuracy: %0.2f (+/- %0.2f)" % (ac.mean(), acc))
#                     print("precision: %0.2f (+/- %0.2f)" % (pr.mean(), prc))
#                     print("recall: %0.2f (+/- %0.2f)" % (rc.mean(), rcl))
#                     print("f1: %0.2f (+/- %0.2f)" % (f1.mean(), f11))
#                     print("specificity: %0.2f (+/- %0.2f)" % (sp.mean(), spc))


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
#                 l=l+1
                print('---------------------------------------------------------------------')
                         
                return
                    
                    
                