# FastFeatGen #

This is a tool for faster feature extraction from genome sequences and making efficient prediction. To build efficient prediction model, user can go through the following instructions step by step. If user only wants to predict query sequences from our built model, then just go to step 'Make prediction for query sequences'.

## System Requirements ##

Users need to have following softwares/tools installed on their PC.
```
GCC version >= 4.9
OpenMP version >= 4.5
Python version >= 3.6.6
```

## Compile and Run feature extraction ##

To extract features from a dataset, type the following command:

```
$ g++ -fopenmp -std=c++11 -O3 featureExtraction.cpp -o featureExtraction 
$ ./featureExtraction -in datasets/dataset.txt -out datasets/extractedfile.csv -gen 2 -nt 4 -w 41 -o 2 -p 4 -g 4

Here,
in - name of input data file that contains positive and negative DNA sequences. 
     All positive sequences are followed by all negative sequences. We shuffle those sequences later.
out - name of output file where extraced features will be saved. A csv file name is recommended.
gen - An integer value to denote genome. 2 for rice genome, 3 for rat genome. 0 (or 1) for only feature
     construction from rice (or rat) genome if Positive.csv and Negative.csv files exist in datasets folder.
nt - number of threads to be used to extract features.
w - window size of each DNA sequence.
o - order of position independent features. This is also known as k-mer or n-gram.
p - number of positions to be considered for order $o$-mers while constructing position specific features.
g - number of gap to be considered between two DNA alphabets while constructing di-gapped features. 
```

Extracted features file 'extractedfile.csv' will be saved in datasets directory.

## Run model generations ##

To build best model from SVM, RF or ETC, run the following:
```
$ python fastfeatgen.py -f datasets/extractedfile.csv -m 1 -p datasets/Positive.txt -n datasets/Negative.txt

Here,
f - name of extracted feature file.
m - an integer to denote rice (1) or rat(2) genome in output file.
p - name of file that contains positive sequences (one sequence per line).
n - name of file that contains negative sequences (one sequence per line).
```

## Make prediction for query sequences ##

To predict classification for a set of query sequences, use the following command:
```
$ python predict.py -q datasets/queryfile.txt -b results/models/bestModelETC10g1.joblib -f results/models/importantfeaturesETC10g1.txt -nt 4

Here,
q - name of query file where all query sequences are present one per line.
b - name of model file based on which prediction will be made.
f - name of important feature files for provided model file.
nt - number of threads to be used to extract features from query sequences.
```
After running this, our provided model will be used to predict the classification of query sequences.

## Extracted Features and Results ##
Some extracted features can be found [here](https://drive.google.com/drive/folders/1X8qFrkBZM-Anzt3Z8U-duK0R313pLfSA?usp=sharing). Our results and log files can be found at `results` folder.

## Contact
This tool is maintained by Md. Khaledur Rahman. If you face any problem or have any query, don't hesitate to send an email to khaled.cse.07@gmail.com or, morahma@iu.edu
