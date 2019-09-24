import os, subprocess
import pandas as pd
from joblib import load
import argparse
import time

def extractQueryFeatures(sequencefile, nt, opt):
        print("Please wait while FastFeatGen is working .....")
        start = time.time()
        command = "./featureExtraction -in "+ str(sequencefile) + " -out queryFeatures.csv -gen "+ str(opt) +" -nt "+ str(nt) +" -w 41 -o 5 -p 30 -g 28"
        subp = os.popen(command)
        subp.read()
        subp.close()
        end = time.time()
        print("Construction of features from query sequences is complete!!")
        print("Required time:", end - start)

def readFeatures(impfeaturefile):
        print("Generating important features based dataset for query sequences!!")
        importantfeaturefile = open(impfeaturefile)
        importantFeatures = importantfeaturefile.readline().strip().split(",")[1:]
        importantfeaturefile.close()
        Xtest = pd.read_csv("queryFeatures.csv", sep=',')
        features_not_present = list(set(importantFeatures) - set(list(Xtest.columns.values)))
        Xtest = pd.concat([Xtest, pd.DataFrame(columns = features_not_present)])
        Xtest[features_not_present] = [0] * len(features_not_present)
        Xtest = Xtest[importantFeatures]
        return Xtest

def predictbyModel(Xtest, modelfile):
        print("Now making predictions...",)
        model = load(modelfile) 
        predictedvalues = list(model.predict(Xtest))
        print("Done!")
        return predictedvalues

if __name__=="__main__":
        parser = argparse.ArgumentParser(description='Evaluation by FastFeatGen', add_help=True)
        parser.add_argument('-q', '--q', required=True, type=str, help='Filename of query sequences. All DNA sequences should be in a txt file.')
        parser.add_argument('-b', '--b', required=True, type=str, help='Model file of SVM or ETC')
        parser.add_argument('-f', '--f', required=True, type=str, help='Important features file of SVM or ETC')
        parser.add_argument('-nt', '--nt', required=True, type=int, help='Number of threads to consider. It should be less than or equal to number of available cores.')
        args = parser.parse_args()
        qfile = args.q
        nt = args.nt
        mod = args.b
        feat = args.f
        if os.path.isfile(qfile):
            opt = 0
            if "g2" in mod:
                opt = 1
            extractQueryFeatures(qfile, nt, opt)
            Xtest = readFeatures(feat)
            predictions = predictbyModel(Xtest, mod)
            seqfile = open(qfile)
            i = 0
            print("Predictions:")
            for line in seqfile.readlines():
                print(line.strip(), "--->", predictions[i])
                i += 1
            seqfile.close()
        else:
            print("Query file not found!!!")
