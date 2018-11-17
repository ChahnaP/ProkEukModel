#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:09:11 2018

@author: ChahnaPatel
"""
import argparse
from random import randint
#from Bio import SeqIO
#from Bio import Entrez
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression, chi2, mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import csv
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd



def logreg(numoffeatures,featureselectionmethod):
    h = open("scikittablogregfileamarelselfcrossvalavg2copy.tab", "a")
    g = open("scikittabfileLogRegavgprf12copy.tab","a")

    #plt.figure(1)
    Xnewarray = []
    ynewarray = []
    X = np.array(listoflistfeatures).astype(np.float)
    X1 = listoflistfeatures1
    X2 = listoflistfeatures2
    X3 = listoflistfeatures3
    X4 = listoflistfeatures4
    X5 = listoflistfeatures5
    X6 = listoflistfeatures6
    X7 = listoflistfeatures7
    X8 = listoflistfeatures8
    X9 = listoflistfeatures9
    X10 = listoflistfeatures10
    X11 = np.array(listoflistfeatures11).astype(np.float)
    y = np.array(listclass).astype(np.float)
    y1 = listclass1
    y2 = listclass2
    y3 = listclass4
    y4 = listclass4
    y5 = listclass5
    y6 = listclass6
    y7 = listclass7
    y8 = listclass8
    y9 = listclass9
    y10 = listclass10
    y11 = np.array(listclass11).astype(np.float)
    #plt.figure(1)
    #plt.clf()
    Xnewarray.append(X1)
    Xnewarray.append(X2)
    Xnewarray.append(X3)
    Xnewarray.append(X4)
    Xnewarray.append(X5)
    Xnewarray.append(X6)
    Xnewarray.append(X7)
    Xnewarray.append(X8)
    Xnewarray.append(X9)
    Xnewarray.append(X10)
    Xnewarrayofficial = np.array(Xnewarray).astype(np.float)
    X_test = X11
    ynewarray.append(y1)
    ynewarray.append(y2)
    ynewarray.append(y3)
    ynewarray.append(y4)
    ynewarray.append(y5)
    ynewarray.append(y6)
    ynewarray.append(y7)
    ynewarray.append(y8)
    ynewarray.append(y9)
    ynewarray.append(y10)
    ynewarrayofficial = np.array(ynewarray).astype(np.float)
    y_test = y11
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.34)
    kf = KFold(n_splits = 10, shuffle = True)
    #kf.get_n_splits(X)
    #KFold(n_splits = 10, random_state = None, shuffle = False)
    my_dict = {"precision": [], "recall": [], "f1-score": [], "class": [], "numoffeatures": [], "featselmethod": [] }
    
    tncount = 0
    fpcount = 0
    fncount = 0
    tpcount = 0
    for q in range(0,10):
        X_train = Xnewarrayofficial[q]
        
        y_train = ynewarrayofficial[q]
        
        selector = SelectKBest(featureselectionmethod, k=numoffeatures)
        selector.fit(X_train, y_train)
        X_indices = np.arange(X_train.shape[-1])
    # #############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
        selector = SelectKBest(featureselectionmethod, k=numoffeatures)
        #selector.fit(X_train, y_train)
            #scores = selector.pvalues_
            #scores /= scores.max()
            #plt.bar(X_indices - .45, selector, width=.2,label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',edgecolor='black')
            
            # #############################################################################
            # Compare to the weights of an SVM
        #logit_model = sm.Logit(y_train,X_train)
        #result = logit_model.fit()
            
        #svm_weights = (clf.coef_ ** 2).sum(axis=0)
        #svm_weights /= svm_weights.max()
        
        
        #plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
                        #color='navy', edgecolor='black')
        logreg = LogisticRegression()  
        #logreg_model_selected = logreg.fit(selector.transform(X_train),y_train)
        #print(logreg_model_selected)
        feature_logreg = make_pipeline(selector,logreg)
        feature_logreg.fit(X_train,y_train)
        y_pred_class = feature_logreg.predict(X_train)
        y_true = y_train
        y_pred = y_pred_class
        print(classification_report(y_true,y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
        print(str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp))
        tncount += tn
        fpcount += fp
        fncount += fn
        tpcount += tp
#        for i in range(1,3):
#            print(precision_recall_fscore_support(y_true, y_pred, average = 'binary',pos_label = i))
#            prfarray = precision_recall_fscore_support(y_true, y_pred, average = 'binary',pos_label = i)
#            print(prfarray)
#            print(prfarray[0])
#            h.write(str(prfarray[0]) + "\t" + str(prfarray[1]) + "\t" + str(prfarray[2]) + "\t" + str(i) + "\t" + str(numoffeatures) + "\t" + featureselectionmethod.__name__ + "\n")
        prfarrayweighted = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
        print(prfarrayweighted)
        h.write(str(prfarrayweighted[0]) + "\t" + str(prfarrayweighted[1]) + "\t" + str(prfarrayweighted[2]) + "\t" + str(3) + "\t" + str(numoffeatures) + "\t" + featureselectionmethod.__name__ + "\n")
    tnavg = tncount/10
    fpavg = fpcount/10
    fnavg = fncount/10
    tpavg = tpcount/10
    print("TNavg" + str(tnavg))
    print("FPavg" + str(fpavg))
    print("FNavg" + str(fnavg))
    print("TPavg" + str(tpavg))
    precision = tpavg/(tpavg + fpavg)
    recall = tpavg/(tpavg + fnavg)
    f1score = 2*((precision*recall)/(precision + recall))
    print("Precision" + str(precision) + "Recall" + str(recall) + "F1Score" + str(f1score))
    g.write(str(precision) + "\t" + str(recall) + "\t" + str(f1score) + "\t" + str(3) + "\t" + str(numoffeatures) + "\t" + featureselectionmethod.__name__ + "\n")
    g.close()
   
    h.close()
       

def SVM(numoffeatures,featureselectionmethod):
    Xnewarray = []
    ynewarray = []
    X = np.array(listoflistfeatures).astype(np.float)
    X1 = listoflistfeatures1
    X2 = listoflistfeatures2
    X3 = listoflistfeatures3
    X4 = listoflistfeatures4
    X5 = listoflistfeatures5
    X6 = listoflistfeatures6
    X7 = listoflistfeatures7
    X8 = listoflistfeatures8
    X9 = listoflistfeatures9
    X10 = np.array(listoflistfeatures10).astype(np.float)
    y = np.array(listclass).astype(np.float)
    y1 = listclass1
    y2 = listclass2
    y3 = listclass4
    y4 = listclass4
    y5 = listclass5
    y6 = listclass6
    y7 = listclass7
    y8 = listclass8
    y9 = listclass9
    y10 = np.array(listclass10).astype(np.float)
    #plt.figure(1)
    #plt.clf()
    Xnewarray.append(X1)
    Xnewarray.append(X2)
    Xnewarray.append(X3)
    Xnewarray.append(X4)
    Xnewarray.append(X5)
    Xnewarray.append(X6)
    Xnewarray.append(X7)
    Xnewarray.append(X8)
    Xnewarray.append(X9)
    Xnewarrayofficial = np.array(Xnewarray).astype(np.float)
    print(Xnewarrayofficial)
    X_test = X10
    ynewarray.append(y1)
    ynewarray.append(y2)
    ynewarray.append(y3)
    ynewarray.append(y4)
    ynewarray.append(y5)
    ynewarray.append(y6)
    ynewarray.append(y7)
    ynewarray.append(y8)
    ynewarray.append(y9)
    ynewarrayofficial = np.array(ynewarray).astype(np.float)
    print(ynewarrayofficial)
    y_test = y10
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.34)
#    kf = KFold(n_splits = 10, shuffle = True)
    #kf.get_n_splits(X)
    #KFold(n_splits = 10, random_state = None, shuffle = False)
    h = open("scikittabfilesvmtestSVR.tab", "a")
#    mydict = {"precision": [], "recall": [], 
#               "f1-score": [], "class": [], 
#               "numoffeatures": [], "featselmethod": [] }
#    for train_index, test_index in kf.split(X):
#        prfarray = []
#        prfarrayweighted = []
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = y[train_index], y[test_index]
#        print ("h")
#        X_indices = np.arange(X_train.shape[-1])
    # #############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
    for q in range(0,10):
        X_train = Xnewarrayofficial[q]
        print(X_train)
        y_train = ynewarrayofficial[q]
        print(y_train)
        selector = SelectKBest(featureselectionmethod, k=numoffeatures)
        selector.fit(X_train, y_train)
            #scores = selector.pvalues_
            #scores /= scores.max()
            #plt.bar(X_indices - .45, selector, width=.2,label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',edgecolor='black')
            
            # #############################################################################
            # Compare to the weights of an SVM
        clf = svm.SVC(kernel='rbf')
        clf.fit(X_train, y_train)
            
#        svm_weights = (clf.coef_ ** 2).sum(axis=0)
#        svm_weights /= svm_weights.max()
#        
#        
#        plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
#                        color='navy', edgecolor='black')
#                
        clf_selected = svm.SVC(kernel='rbf')
        clf_selected.fit(selector.transform(X_train), y_train)
        feature_svm = make_pipeline(selector,clf_selected)
        y_pred_class = feature_svm.predict(X_test)
        y_true = y_test
        y_pred = y_pred_class
        print(classification_report(y_true,y_pred))
#        for i in range(1,3):
#            print(precision_recall_fscore_support(y_true, y_pred, average = 'binary',pos_label = i))
#            prfarray = precision_recall_fscore_support(y_true, y_pred, average = 'binary',pos_label = i)
#            print(prfarray)
#            print(prfarray[0])
#            h.write(str(prfarray[0]) + "\t" + str(prfarray[1]) + "\t" + str(prfarray[2]) + "\t" + str(i) + "\t" + str(numoffeatures) + "\t" + featureselectionmethod.__name__ + "\n")
#            mydict["precision"].append(prfarray[0])
#            mydict["recall"].append(prfarray[1])
#            mydict["f1-score"].append(prfarray[2])
#            mydict["class"].append(i)
#            mydict["numoffeatures"].append(numoffeatures)
#            mydict["featselmethod"].append(featureselectionmethod.__name__)
        prfarrayweighted = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
#        mydict["precision"].append(prfarrayweighted[0])
#        mydict["recall"].append(prfarrayweighted[1])
#        mydict["f1-score"].append(prfarray[2])
#        mydict["class"].append(3)
#        mydict["numoffeatures"].append(numoffeatures)
#        mydict["featselmethod"].append(featureselectionmethod.__name__)
        print(prfarrayweighted)
        h.write(str(prfarrayweighted[0]) + "\t" + str(prfarrayweighted[1]) + "\t" + str(prfarrayweighted[2]) + "\t" + str(3) + "\t" + str(numoffeatures) + "\t" + featureselectionmethod.__name__ + "\n") 
    h.close()
    
    
def RandomForest():
    Xnewarray = []
    ynewarray = []
    X = np.array(listoflistfeatures).astype(np.float)
    X1 = listoflistfeatures1
    X2 = listoflistfeatures2
    X3 = listoflistfeatures3
    X4 = listoflistfeatures4
    X5 = listoflistfeatures5
    X6 = listoflistfeatures6
    X7 = listoflistfeatures7
    X8 = listoflistfeatures8
    X9 = listoflistfeatures9
    X10 = listoflistfeatures10
    X11 = np.array(listoflistfeatures11).astype(np.float)
    y = np.array(listclass).astype(np.float)
    y1 = listclass1
    y2 = listclass2
    y3 = listclass4
    y4 = listclass4
    y5 = listclass5
    y6 = listclass6
    y7 = listclass7
    y8 = listclass8
    y9 = listclass9
    y10 = listclass10
    y11 = np.array(listclass11).astype(np.float)
    #plt.figure(1)
    #plt.clf()
    Xnewarray.append(X1)
    Xnewarray.append(X2)
    Xnewarray.append(X3)
    Xnewarray.append(X4)
    Xnewarray.append(X5)
    Xnewarray.append(X6)
    Xnewarray.append(X7)
    Xnewarray.append(X8)
    Xnewarray.append(X9)
    Xnewarray.append(X10)
    Xnewarrayofficial = np.array(Xnewarray).astype(np.float)
    print(Xnewarrayofficial)
    #datax = pd.DataFrame(Xnewarrayofficial)
    dfx = pd.DataFrame.from_records(X, columns = ['ATContent', 'AAA', 'CCC', 'TTT', 'GGG', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA', 'TGG', 'CTT','CTC','CTA','CTG', 'CCT', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA'])
    #print(Xnewarrayofficial)
    X_test = X11
    ynewarray.append(y1)
    ynewarray.append(y2)
    ynewarray.append(y3)
    ynewarray.append(y4)
    ynewarray.append(y5)
    ynewarray.append(y6)
    ynewarray.append(y7)
    ynewarray.append(y8)
    ynewarray.append(y9)
    ynewarray.append(y10)
    ynewarrayofficial = np.array(ynewarray).astype(np.float)
    print(ynewarrayofficial)
    #datay = pd.DataFrame(ynewarrayofficial)
    #dfy = pd.DataFrame.from_records(y, columns = ['ATContent','AAA', 'CCC', 'TTT', 'GGG', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA', 'TGG', 'CTT','CTC','CTA','CTG', 'CCT', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA'])
    #print(ynewarrayofficial)
    y_test = y11
    
    #plt.figure(1)
    #plt.clf()
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.34)
    #kf.get_n_splits(X)
    #KFold(n_splits = 10, random_state = None, shuffle = False)
    h = open("scikittabfileRandomForestselfcrossvalavg2copy.tab", "a")
    g = open("scikittabfileRandomForestavgprf12copy.tab","a")
    i = open("scikittabfileRandomForestfeatureimportance.tab", "a")
    
    tncount = 0
    fpcount = 0
    fncount = 0
    tpcount = 0
    # #############################################################################
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 10% most significant features
    for r in range(0,9):
        X_train = Xnewarrayofficial[r]
        #print(X_train)
        y_train = ynewarrayofficial[r]
        #print(y_train)
#        selector = SelectKBest(featureselectionmethod, k=numoffeatures)
#        selector.fit(X_train, y_train)
#            #scores = selector.pvalues_
            #scores /= scores.max()
            #plt.bar(X_indices - .45, selector, width=.2,label=r'Univariate score ($-Log(p_{value})$)', color='darkorange',edgecolor='black')
            
            # #############################################################################
            # Compare to the weights of an SVM
        randomForest = RandomForestClassifier(max_features = 22, min_samples_leaf = 5, n_estimators = 100)
        
            
#        svm_weights = (clf.coef_ ** 2).sum(axis=0)
#        svm_weights /= svm_weights.max()
#        
#        
#        plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
#                        color='navy', edgecolor='black')
#                

#        feature_randomForest = make_pipeline(selector,randomForest)
        randomForest.fit(X_train,y_train)
        y_pred_class = randomForest.predict(X_train)
        y_true = y_train
        y_pred = y_pred_class
        print(classification_report(y_true,y_pred))
        tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
        print(str(tn) + " " + str(fp) + " " + str(fn) + " " + str(tp))
        tncount += tn
        fpcount += fp
        fncount += fn
        tpcount += tp
#        for i in range(1,3):
#            print(precision_recall_fscore_support(y_true, y_pred, average = 'binary',pos_label = i))
#            prfarray = precision_recall_fscore_support(y_true, y_pred, average = 'binary',pos_label = i)
#            print(prfarray)
#            print(prfarray[0])
#            h.write(str(prfarray[0]) + "\t" + str(prfarray[1]) + "\t" + str(prfarray[2]) + "\t" + str(i) + "\t" + str(numoffeatures) + "\t" + featureselectionmethod.__name__ + "\n")
#            mydict["precision"].append(prfarray[0])
#            mydict["recall"].append(prfarray[1])
#            mydict["f1-score"].append(prfarray[2])
#            mydict["class"].append(i)
#            mydict["numoffeatures"].append(numoffeatures)
#            mydict["featselmethod"].append(featureselectionmethod.__name__)
        prfarrayweighted = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
#        mydict["precision"].append(prfarrayweighted[0])
#        mydict["recall"].append(prfarrayweighted[1])
#        mydict["f1-score"].append(prfarray[2])
#        mydict["class"].append(3)
#        mydict["numoffeatures"].append(numoffeatures)
#        mydict["featselmethod"].append(featureselectionmethod.__name__)
        print(prfarrayweighted)
        h.write(str(prfarrayweighted[0]) + "\t" + str(prfarrayweighted[1]) + "\t" + str(prfarrayweighted[2]) + "\t" + str(3) + "\n")
    tnavg = tncount
    fpavg = fpcount
    fnavg = fncount
    tpavg = tpcount
    print("TNavg" + str(tnavg))
    print("FPavg" + str(fpavg))
    print("FNavg" + str(fnavg))
    print("TPavg" + str(tpavg))
    precision = tpavg/(tpavg + fpavg)
    recall = tpavg/(tpavg + fnavg)
    f1score = 2*((precision*recall)/(precision + recall))
    print("Precision " + str(precision) + "Recall" + str(recall) + "F1Score" + str(f1score))
    featureimportances = pd.DataFrame(randomForest.feature_importances_, index = dfx.columns, columns = ['importance']).sort_values('importance',ascending = False)
    print(featureimportances)
    stringfeatureimportances = featureimportances.to_string();
    headers = ['ATContent', 'AAA', 'CCC', 'TTT', 'GGG', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA', 'TGG', 'CTT','CTC','CTA','CTG', 'CCT', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC', 'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA']
    i.write(stringfeatureimportances)
    #featureimportances = sorted(zip(X.columns, randomForest.feature_importances_), key=lambda x: x[1] * -1)
    #print(tabulate(featureimportances, headers, tablefmt="plain"))
    g.write(str(precision) + "\t" + str(recall) + "\t" + str(f1score) + "\t" + str(3) + "\n")
    g.close()
    h.close()
    i.close()
        
    
        
def main():
        """
        This is your main function
        This will be the function that will be exectued if you run the
        script from commandline
        """
listoflistfeatures = []
with open('newsmallprokeuk1shuf.tab','r') as f:
    reader = csv.reader(f, delimiter = '\t')
    for row in reader:
        listfeatures = []
        for i in range(len(row)-1):
            if len(row) == 66:
            #print (row[19])
                listfeatures.append((row[i]))
        #print(listfeatures)
        listoflistfeatures.append(listfeatures)
        #print(listoflistfeatures)
print(listoflistfeatures)
      
listoflistfeatures1 = []
for o1 in range(0,67500):
    listoflistfeatures1.append(listoflistfeatures[o1])
#print(listoflistfeatures2)
    
listoflistfeatures2 = []
for o2 in range(67500,135000):
    listoflistfeatures2.append(listoflistfeatures[o2])
#print(listoflistfeatures3)
    
listoflistfeatures3 = []
for o3 in range(135000,202500):
    listoflistfeatures3.append(listoflistfeatures[o3])
#print(listoflistfeatures4)
    
listoflistfeatures4 = []
for o4 in range(202500,270000):
    listoflistfeatures4.append(listoflistfeatures[o4])
#print(listoflistfeatures4)
 
listoflistfeatures5 = []    
for o5 in range(270000,337500):
    listoflistfeatures5.append(listoflistfeatures[o5])
#print(listoflistfeatures4)

listoflistfeatures6 = []    
for o6 in range(337500,405000):
    listoflistfeatures6.append(listoflistfeatures[o6])
#print(listoflistfeatures4)
    
listoflistfeatures7 = []    
for o7 in range(405000,472500):
    listoflistfeatures7.append(listoflistfeatures[o7])
#print(listoflistfeatures4)

listoflistfeatures8 = []    
for o8 in range(472500,540000):
    listoflistfeatures8.append(listoflistfeatures[o8])
#print(listoflistfeatures4)    
   
listoflistfeatures9 = []    
for o9 in range(540000,607500):
    listoflistfeatures9.append(listoflistfeatures[o9])
#print(listoflistfeatures4)   

listoflistfeatures10 = []    
for o10 in range(607500,675000):
    listoflistfeatures10.append(listoflistfeatures[o10])
#print(listoflistfeatures4)  
    
listoflistfeatures11 = []
for o11 in range(675000,742500):
    listoflistfeatures11.append(listoflistfeatures[o11])

    
with open('newsmallprokeuk1shuf.tab','r') as f:
    listclass = []
    reader = csv.reader(f, delimiter = '\t')
    for row in reader:
        if len(row) == 66:
        #print (row[19])
            listclass.append(row[65])
    #print(listclass)


listclass1 = []
for p1 in range(0,67500):
    listclass1.append(listclass[p1])
#print(listclass2)

listclass2 = []
for p2 in range(67500,135000):
    listclass2.append(listclass[p2])
#print(listclass3)

listclass3 = []
for p3 in range(135000,202500):
    listclass3.append(listclass[p3])
#print(listclass4)

listclass4 = []
for p4 in range(202500,270000):
    listclass4.append(listclass[p4])
#print(listclass4)

listclass5 = []
for p5 in range(270000,337500):
    listclass5.append(listclass[p5])
#print(listclass4)

listclass6 = []
for p6 in range(337500,405000):
    listclass6.append(listclass[p6])
#print(listclass4)

listclass7 = []
for p7 in range(405000,472500):
    listclass7.append(listclass[p7])
#print(listclass4)

listclass8 = []
for p8 in range(472500,540000):
    listclass8.append(listclass[p8])
#print(listclass4)

listclass9 = []
for p9 in range(540000,607500):
    listclass9.append(listclass[p9])
#print(listclass4)

listclass10 = []
for p10 in range(607500,675000):
    listclass10.append(listclass[p10])
#print(listclass4)

listclass11 = []
for p11 in range(675000,742500):
    listclass11.append(listclass[p11])
    

listnumfeatures = []
for i in range(1,65):
    listnumfeatures.append(i)    
#listfeatureselmethod = ["chi2", "f_classif", "mutual_info_classif", "f_regression", "mutual_info_regression"]
listfeatureselmethod = [chi2, f_classif, mutual_info_classif, f_regression, mutual_info_regression]
parser = argparse.ArgumentParser()
parser.add_argument("function", nargs = "?", choices = ['logreg','svm','randomforest'], default = 'logreg')
#parser.add_argument("featureselectionmethod", nargs = "?",choices = ["chi2", "f_classif", "mutual_info_classif", "f_regression", "mutual_info_regression"], default = "f_regression")
#parser.add_argument('numoffeatures', type=int, default=1)
args, sub_args = parser.parse_known_args()

#if sys.version[0] == "3": 
#    raw_input = input
#    
#s = raw_input("What feature selection method do you want to test?")
#if args.function == "logreg":
#    print("logreg")
#    if args.featureselectionmethod == "chi2":
#        print("chi2")
#        logreg(args.numoffeatures,listfeatureselmethod[0])
#    elif args.featureselectionmethod == "f_classif":
#        print("f_classif")
#        logreg(args.numoffeatures,listfeatureselmethod[1])
#    elif args.featureselectionmethod == "mutual_info_classif":
#        logreg(args.numoffeatures,listfeatureselmethod[2])
#        print("mutual_info_classif")
#    elif args.featureselectionmethod == "f_regression":
#        logreg(args.numoffeatures,listfeatureselmethod[3])
#        print("f_regression")
#    elif args.featureselectionmethod == "mutual_info_regression":
#        logreg(args.numoffeatures,listfeatureselmethod[4])
#        print("mutual_info_regression")
#elif args.function == "svm":
#    print("svm")
#    if args.featureselectionmethod == "chi2":
#        SVM(args.numoffeatures,listfeatureselmethod[0])
#        print("chi2")
#    elif args.featureselectionmethod == "f_classif":
#        SVM(args.numoffeatures,listfeatureselmethod[1])
#        print("f_classif")
#    elif args.featureselectionmethod == "mutual_info_classif":
#        SVM(args.numoffeatures,listfeatureselmethod[2])
#        print("mutual_info_classif")
#    elif args.featureselectionmethod == "f_regression":
#        SVM(args.numoffeatures,listfeatureselmethod[3])
#        print("f_regression")
#    elif args.featureselectionmethod == "mutual_info_regression":
#        SVM(args.numoffeatures,listfeatureselmethod[4])
#        print("mutual_info_regression")

if args.function == "logreg":
    for i in listfeatureselmethod:
        for j in range(1,66):
            logreg(j,i)
            print(str(j))
            print("logreg")
            
elif args.function == "svm":
    for k in listfeatureselmethod:
        for l in range(1,66):
            SVM(l,k)
            print(str(l))
            print("svm")

elif args.function == "randomforest":
    RandomForest()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
