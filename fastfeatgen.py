import os
import sys
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import argparse
import math
from joblib import dump
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesClassifier
from random import shuffle
import warnings
warnings.filterwarnings("ignore")

def featureSelectionRF(X, Y):
    print("Total extracted features:", len(X[1,]))
    forest = RandomForestClassifier(max_depth=500)
    forest.fit(X, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    print("Creating RF feature ranking.....")
    importantIndex = []
    for f in range(X.shape[1]):
        if importances[indices[f]] > 0:
            importantIndex.append(indices[f])
        else:
            break
    print("Number of important RF features:", len(importantIndex))
    return importantIndex

def writeimportantFeatures(f, fi, filename):
    featurefile = open(filename, "w")
    #featurefile.write("[")
    for i in fi:
        featurefile.write(","+ f[i])
    #featurefile.write("]\n")
    featurefile.close()

def featureSelectionSVC(X, Y):
    print("Total extracted features:", len(X[1,]))
    svm = LinearSVC()
    svm.fit(X, Y)
    coef = svm.coef_.ravel()
    # print(coef.ravel()[0:10])
    print("SVC coef length:", len(coef))
    print("Creating SVC feature ranking.....")
    importantIndex = []
    for f in range(len(coef)):
        if abs(coef[f]) > 0.01:
            importantIndex.append(f)
    print("Number of important SVC features:", len(importantIndex))
    return importantIndex

def crossValidation(X, Y, cv, features, m):
    bestAccuracy = 0
    bestModel = None
    importantfeatures = []
    dindex = list(range(0, len(Y)))
    shuffle(dindex)
    X = X[dindex]
    Y = Y[dindex]
    indices = featureSelectionRF(X, Y)
    indicesSVM = featureSelectionSVC(X, Y)
    
    lSVC = LinearSVC(C=0.1, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lSVC, prefit=True)
    X_svm = model.transform(X)
    CV = cv
    allindices = list(range(0, len(Y)))
    trueY = []
    SVMpredictedY = []
    for iterations in range(CV):
        cvindices = []
        for c in range(int(math.ceil(len(Y))/CV)):
            cvindices.append(allindices.pop(0))
        Xtrain = X_svm[allindices]
        Ytrain = Y[allindices]
        Xtest = X_svm[cvindices]
        Ytest = Y[cvindices]
        trueY = trueY + list(Ytest)
        clfSVC = SVC(kernel='linear', C = 0.1)
        clfSVC.fit(Xtrain, Ytrain)
        SVMpredictedY = SVMpredictedY + list(clfSVC.predict(Xtest))
        allindices = allindices + cvindices
    tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(trueY, SVMpredictedY).ravel()

    print('SVM Accuracy Initial:', accuracy_score(trueY, SVMpredictedY), ' MCC Initial:', matthews_corrcoef(trueY, SVMpredictedY))
    print ('SVM Sensitivity Initial:', (tp_svm)*1.0/(tp_svm + fn_svm), ' Specificity Initial:', (tn_svm)*1.0/(tn_svm + fp_svm))
    dump(clfSVC, "results/models/bestModelSVMwithAll"+str(cv)+"g"+ str(m)+".joblib")
    allindices = list(range(0, len(Y)))
    writeimportantFeatures(features, allindices, "results/models/importantfeaturesSVMwithAll"+str(cv)+"g"+ str(m)+".txt")
    CV = 10
    for i in range(len(indicesSVM) - 1, 1200, -130):
        print("Working for ",i, " SVM features......")
        Xt = X[:,indicesSVM[0:i]]
        allindices = list(range(0, len(Y)))
        trueY = []
        SVMpredictedY = []
        for iterations in range(CV):
            cvindices = []
            for c in range(int(math.ceil(len(Y))/CV)):
                cvindices.append(allindices.pop(0))
            Xtrain = Xt[allindices]
            Ytrain = Y[allindices]
            Xtest = Xt[cvindices]
            Ytest = Y[cvindices]
            trueY = trueY + list(Ytest)
            clfSVC = SVC(kernel='linear', C = 0.1)
            clfSVC.fit(Xtrain, Ytrain)
            SVMpredictedY = SVMpredictedY + list(clfSVC.predict(Xtest))
            allindices = allindices + cvindices
        tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(trueY, SVMpredictedY).ravel()
        print('SVM Accuracy:', accuracy_score(trueY, SVMpredictedY), ' MCC:', matthews_corrcoef(trueY, SVMpredictedY))
        print ('SVM Sensitivity:', (tp_svm)*1.0/(tp_svm + fn_svm), ' Specificity:', (tn_svm)*1.0/(tn_svm + fp_svm))
        if accuracy_score(trueY, SVMpredictedY) > bestAccuracy:
            bestAccuracy = accuracy_score(trueY, SVMpredictedY)
            bestModel = clfSVC
            importantfeatures = indicesSVM[0:i]
            print("\n\n.............BEST ACCURACY from SVM..............\n\n")
    dump(bestModel, "results/models/bestModelSVM"+str(cv)+"g"+ str(m)+".joblib")
    writeimportantFeatures(features, importantfeatures, "results/models/importantfeaturesSVM"+str(cv)+"g"+ str(m)+".txt")
    bestAccuracy = 0
    importantfeatures = []
    CV = cv
    for i in range(len(indices) - 1, 100, -25):
        print("Working for ",i, " RF features......")
        Xt = X[:,indices[0:i]]
        allindices = list(range(0, len(Y)))
        trueY = []
        ETCpredictedY = []
        for iterations in range(CV):
            cvindices = []
            for c in range(int(math.ceil(len(Y))/CV)):
                cvindices.append(allindices.pop(0))
            Xtrain = Xt[allindices]
            Ytrain = Y[allindices]
            Xtest = Xt[cvindices]
            Ytest = Y[cvindices]
            trueY = trueY + list(Ytest)
            clfETC = ExtraTreesClassifier(max_depth=200)
            clfETC.fit(Xtrain, Ytrain)
            ETCpredictedY = ETCpredictedY + list(clfETC.predict(Xtest))
            allindices = allindices + cvindices
        tn_etc, fp_etc, fn_etc, tp_etc = confusion_matrix(trueY, ETCpredictedY).ravel()
        print ('ETC Accuracy:', accuracy_score(trueY, ETCpredictedY), ' MCC:', matthews_corrcoef(trueY, ETCpredictedY))
        print ('ETC Sensitivity:', (tp_etc)*1.0/(tp_etc + fn_etc), ' Specificity:', (tn_etc)*1.0/(tn_etc + fp_etc))
        if accuracy_score(trueY, ETCpredictedY) > bestAccuracy:
            bestAccuracy = accuracy_score(trueY, ETCpredictedY)
            bestModel = clfETC
            importantfeatures = indices[0:i]
            print("\n\n.............BEST ACCURACY from ETC..............\n\n")
    dump(bestModel, "results/models/bestModelETC"+str(cv)+"g"+ str(m)+".joblib")
    writeimportantFeatures(features, importantfeatures, "results/models/importantfeaturesETC"+str(cv)+"g"+ str(m)+".txt")
    print("\n")
    return importantfeatures

def readFeaturizedData(fullData, posData, negData):
    fulldata = pd.read_csv(fullData, sep=',')
    featurenames = list(fulldata.columns.values)
    X = np.array(fulldata)
    print("Length X = ", len(X))
    #assert(np.any(np.isnan(X)))
    #assert(np.any(np.isinf(X)))
    dY1 = np.array(pd.read_csv(posData, sep=','))
    Y1 = np.full(len(dY1)+1, 1)
    print("Length Y1 = ", len(Y1))
    #assert(np.isnan(Y1))
    dY2 = np.array(pd.read_csv(negData, sep=','))
    Y2 = np.full(len(dY2)+1, 0)
    print("Length Y2 = ", len(Y2))
    Y = np.concatenate((Y1, Y2), axis = 0)
    assert(len(X) == len(Y))
    return X, Y, featurenames

def main(ffile, pos, neg, m):
    FullData = ffile
    posFile = pos
    negFile = neg
    x, y, f = readFeaturizedData(FullData, posFile, negFile)
    cvs = [10]
    for cv in cvs:
        print("\n\nGenerating results for ", cv, " crossvalidations...\n\n\n\n")
        fi = crossValidation(x, y, cv, f, m)
    #featurefile = open("importantfeatures.txt", "w")
    #featurefile.write("[")
    #for i in fi:
    #    featurefile.write(","+ f[i])
    #featurefile.write("]\n")
    #featurefile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Building Model by  FastFeatGen', add_help=True)
    parser.add_argument('-m', '--m', required=True, type=int, help='An integer to denote genome name. 1 for Rice Genome, 2 for rat genome.')
    parser.add_argument('-f', '--f', required=True, type=str, help='Filename of extracted features.')
    parser.add_argument('-p', '--p', required=True, type=str, help='Text file that contains positive sequences.')
    parser.add_argument('-n', '--n', required=True, type=str, help='Text file that contains negative sequences.')
    args = parser.parse_args()
    m = args.m
    ffile = args.f
    pos = args.p
    neg = args.n
    if os.path.isfile(ffile) and os.path.isfile(pos) and os.path.isfile(neg):
        main(ffile, pos, neg, m)
    else:
        print("extracted feature file not found!!")
