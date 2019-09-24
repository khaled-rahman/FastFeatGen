#include <omp.h>
#include <unordered_map>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <iterator>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <random>
#include <utility>

using namespace std;
#define DTYPE float

int ORDER, POSITIONS, GAPPED, gen;
unordered_map<string,vector<DTYPE> > featurematrix; 
void printFeatureMatrix(unordered_map<string, vector<DTYPE> > fm){
        unordered_map<string, vector<DTYPE> >::iterator it = fm.begin();
        while(it != fm.end()){
                cout << it->first << ":";
                vector<DTYPE>::iterator itt = it->second.begin();
                while(itt != it->second.end()){
                        cout << *itt << " ";
                        itt++;
                }
                cout << endl;
                it++;
        }
}

void writeData(unordered_map<string, vector<DTYPE> > fm, int datasize, string filename){
        vector<vector<DTYPE> > dataset(datasize, vector<DTYPE>(fm.size()));
        unordered_map<string, vector<DTYPE> >::iterator it = fm.begin();
        int i = 0, j = 0;
        while(it != fm.end()){
                vector<DTYPE>::iterator itt = it->second.begin();
                i = 0;
                while(itt != it->second.end()){
                        dataset[i][j] = *itt;
                        i++;
                        itt++;
                }
                j++;
                it++;
        }
        ofstream output;
        output.open(filename);
        it = fm.begin();
        output << it->first;
        it++;
        while(it != fm.end()){
                output << "," << it->first;
                it++;
        }
        output << "\n";
        for(int i=0; i<datasize; i++){
                for(int j=0; j<fm.size(); j++){
                        output << to_string(dataset[i][j]);
                        if(j < fm.size() - 1)output << ",";
                }
                output << "\n";
        }
        output.close();
}


void generate_words(string alphabet, int word_length, vector<string> &results){
    vector<int> index(word_length, 0);
    for (;;){
        string word(index.size(), ' ');
        for (int i = 0; i < index.size(); ++i){
            word[i] = alphabet[index[i]];
        }
        results.push_back(word);

        for (int i = index.size() - 1; ; --i){
            if (i < 0){
                return;
            }
            index[i]++;

            if (index[i] == alphabet.size()){
                index[i] = 0;
            }
            else{
                break;
            }
        }
    }
}

int findAllpositions(string s, string sub, vector<int> &positions){
	int flag = 1;
	int index = -1;
	while(flag){
		index = s.find(sub, index+1);
		if(index == -1) flag = 0;
		else positions.push_back(index);
	}
	/*for(int i=0 ;i<positions.size(); i++){
		cout <<"Positions:" <<positions[i] << endl;
	}*/
	return positions.size();
}
void extractPIF(vector<string> data, vector<string> features){
	#pragma omp parallel shared(featurematrix)
	{
		#pragma omp for schedule(static)
		for(int i =0; i< features.size(); i++){
			vector<DTYPE> featurevector(data.size(), 0);
			for(int j=0; j<data.size(); j++){
				vector<int> d;
				int ind = findAllpositions(data[j], features[i], d);
				featurevector[j] = 1.0 * ind / (data[j].size() - features[i].size() + 1);
			}
			/*cout << features[i] << ":";
			for(int j=0; j<featurevector.size(); j++){
				cout << featurevector[j] << " ";
			}
			cout << endl;
			*/
			#pragma omp critical
			{
				if(accumulate(featurevector.begin(), featurevector.end(), 0) > 0)
					featurematrix.insert(make_pair(features[i], featurevector));
			}
		}
	}
}

void extractPSF(vector<string> data, vector<string> features){
	#pragma omp parallel shared(featurematrix)
	{
		#pragma omp for schedule(static)
        	for(int i =0; i< features.size(); i++){
			for(int p = 0; p < POSITIONS; p++){
				string feature = features[i] + "_" + to_string(p);
				vector<DTYPE> featurevector(data.size(), 0);
                		for(int j=0; j<data.size(); j++){
					if(data[j].find(features[i], p) == p){
						featurevector[j] = 1.0;
					}else{
						featurevector[j] = 0.0;
					}		
				}
				#pragma omp critical
				if(accumulate(featurevector.begin(), featurevector.end(), 0) > 0)
					featurematrix.insert(make_pair(feature, featurevector));
                	}
        	}
	}
}

void extractDiGapped(vector<string> data, vector<string> features){	
	#pragma omp parallel shared(featurematrix)
	{
		#pragma omp for schedule(static)
		for(int i=0; i<features.size(); i++){
			for(int g=1; g<= GAPPED; g++){
				vector<DTYPE> featurevector(data.size(), 0);
				string feature = features[i].substr(0,1) + "_" + features[i].substr(1,1) + "_" + to_string(g);
				for(int j=0; j<data.size(); j++){
					vector<int> allpositions;
					findAllpositions(data[j], features[i].substr(0,1), allpositions);
					int count = 0;
					for(int k=0; k<allpositions.size(); k++){
						if(data[j].find(features[i].substr(1,1), allpositions[k] + g) == allpositions[k] + g){
							count++;
						}
					}
					featurevector[j] = 1.0 * count / (data[j].size() - g + 1);
				}
				#pragma omp critical
				if(accumulate(featurevector.begin(), featurevector.end(), 0) > 0)
                               		featurematrix.insert(make_pair(feature, featurevector));
			}
		}
	}
} 

void calcBPP(vector<string> data, vector<string> features, string filename){
	unordered_map<string,vector<DTYPE> > BPP;
	#pragma omp parallel shared(BPP)
        {
                #pragma omp for schedule(static)
                for(int i =0; i< features.size(); i++){
			string feature = features[i];
                        vector<DTYPE> featurevector(data[0].size(), 0);
                        for(int p = 0; p < data[0].size() - features[i].size() + 1; p++){
                                for(int j=0; j<data.size(); j++){
                                        if(data[j].find(features[i], p) == p){
                                                featurevector[p] += 1.0;
                                        }
                                }
				featurevector[p] = (1.0 * featurevector[p] / data.size());
                        }
			#pragma omp critical
                        BPP.insert(make_pair(feature, featurevector));
                }
        }
	//printFeatureMatrix(BPP);
	writeData(BPP, data[0].size(), filename);
}
void extractBPP(vector<string> data, vector<string> features){
	unordered_map<string,vector<DTYPE> > bppMATpos, bppMATneg;
	vector<vector<DTYPE> > temp;
        ifstream positive, negative;
	if(gen == 0){
        	positive.open("datasets/Positive.csv");
        	negative.open("datasets/Negative.csv");
	}else if(gen == 1){
		positive.open("datasets/Positive2.csv");
                negative.open("datasets/Negative2.csv");
	}
	if(positive.is_open()){
                string lines, line;
		getline(positive, lines);
		vector<string> features;
		stringstream f(lines);
		while(getline(f, line, ','))
			features.push_back(line);
                while(getline(positive, lines)){
			vector<DTYPE> d;
			stringstream s(lines);
			while(getline(s, line, ',')){
				d.push_back(stod(line));
			}
			temp.push_back(d);
                }
		for(int j =0; j<temp[0].size(); j++){
			vector<DTYPE> fv;
			for(int i=0; i<temp.size(); i++){
				fv.push_back(temp[i][j]);
			}
			bppMATpos.insert(make_pair(features[j], fv));
		}
        }else{
		cout << "Error while opening file Positive.csv file. No file found!\n";
	}
	temp.clear();
	if(negative.is_open()){
                string lines, line;
                getline(negative, lines);
                vector<string> features;
                stringstream f(lines);
                while(getline(f, line, ','))
                        features.push_back(line);
                while(getline(negative, lines)){
                        vector<DTYPE> d;
                        stringstream s(lines);
                        while(getline(s, line, ',')){
                                d.push_back(stod(line));
                        }
                        temp.push_back(d);
                }
                for(int j =0; j<temp[0].size(); j++){
                        vector<DTYPE> fv;
                        for(int i=0; i<temp.size(); i++){
                                fv.push_back(temp[i][j]);
                        }
                        bppMATneg.insert(make_pair(features[j], fv));
                }
        }else{
                cout << "Error while opening file Negative.csv file. No file found!\n";
        }
	#pragma omp parallel shared(featurematrix)
        {
                #pragma omp for schedule(static)
                for(int i =0; i< features.size(); i++){
                        for(int p = 0; p < data[0].size() - features[i].size() + 1; p++){
                                string feature = features[i] + "_BPP_P_" + to_string(p);
                                vector<DTYPE> featurevector(data.size(), 0);
                                for(int j=0; j<data.size(); j++){
					if(data[j].find(features[i], p) == p){
                                		featurevector[j] = bppMATpos.at(features[i])[p];
					}else{
						featurevector[j] = 0.0;	
					}
                                }
                                #pragma omp critical
                                featurematrix.insert(make_pair(feature, featurevector));
                        }
                }
        }
	#pragma omp parallel shared(featurematrix)
        {
                #pragma omp for schedule(static)
                for(int i =0; i< features.size(); i++){
                        for(int p = 0; p < data[0].size() - features[i].size() + 1; p++){
                                string feature = features[i] + "_BPP_N_" + to_string(p);
                                vector<DTYPE> featurevector(data.size(), 0);
                                for(int j=0; j<data.size(); j++){
                                	if(data[j].find(features[i], p) == p){
                                                featurevector[j] = bppMATneg.at(features[i])[p];
                                        }else{
                                                featurevector[j] = 0.0;                         
                                        }
				}
                                #pragma omp critical
                                featurematrix.insert(make_pair(feature, featurevector));
                        }
                }
        }
}


string checkwsize(string seq, int wsize){
        int lag = seq.size() - wsize;
        string res = "";
        if(lag >= 0){
                for(int i = 0; i < wsize; i++)
                        res += seq[i];
        }else{
                res = seq;
                for(int i = lag; i < 0; i++)
                        res += "X";
        }
        return res;
}

vector<string> readData(string filename, int wsize){
	vector<string> dataset;
	ifstream input;
	input.open(filename);
	if(input.is_open()){
		string line;
		while(input >> line){
			if(line.size() == wsize)
				dataset.push_back(line);
			else
				dataset.push_back(checkwsize(line, wsize));
		}
	}
	input.close();
	return dataset;

}

void featureExtractionDNA(vector<string> data, int nt, string outfile){
	string alphabet = "ACGT";
        vector<string> features;
        for(int i=1; i <= ORDER; i++){
                generate_words(alphabet, i, features);
        }
	vector<string> featuresDiGapped;
        generate_words(alphabet, 2, featuresDiGapped);
	vector<string> featuresBPP;
        generate_words(alphabet, 2, featuresBPP);        
	omp_set_num_threads(nt);
	double start = omp_get_wtime();
	extractPIF(data, features);
	//printf("Size After PIF: %d\n", featurematrix.size());
        extractPSF(data, features);
	//printf("Size After PSF: %d\n", featurematrix.size());
        extractDiGapped(data, featuresDiGapped);
	//printf("Size After DiGapped: %d\n", featurematrix.size());
	extractBPP(data, featuresBPP);
	//printf("Size After BPP: %d\n", featurematrix.size());
	double end = omp_get_wtime();	
	printf("Time required:%lf\n", end - start);
	//printFeatureMatrix();
        writeData(featurematrix, data.size(), outfile);
}

int main(int argc, char* argv[]){
	string fname = "";
	string outputfile = "extractedFeaturesDNA.csv";
	int  nthreads = 1;
	int wsize = 2;
	ORDER = POSITIONS = GAPPED = 2;
	gen = 2;
	for(int i=0; i<argc; i++){
		string arg = argv[i];
		if(arg == "-in"){
			fname = argv[i+1];
			if(fname.size() <= 0){
				printf("No proper input file!!\n");
			}
		}
		if(arg == "-out"){
			outputfile = argv[i+1];
		}
		if(arg == "-gen"){
			gen = atoi(argv[i+1]);
		}
		if(arg == "-nt"){
			nthreads = atoi(argv[i+1]);
		}
		if(arg == "-w"){
			wsize = atoi(argv[i+1]);
		}
		if(arg == "-o"){
			ORDER = atoi(argv[i+1]);
		}
		if(arg == "-p"){
			POSITIONS = atoi(argv[i+1]);
		}
		if(arg == "-g"){
			GAPPED = atoi(argv[i+1]);
		}
	}
	if(POSITIONS > wsize){
		cout << "Positions is greater than a sequence length!\n" << endl;
		return 1;
	}
	
	vector<string> data, datapos, dataneg;
	string alphabet = "ACGT";
        vector<string> featuresBPP;
        generate_words(alphabet, 2, featuresBPP);
	data = readData(fname, wsize);
	if(gen == 2){
		cout << "Wait... Creating BPP Tables for positive and negative datasets...";
		datapos = readData("datasets/Positive.txt", wsize);
		dataneg = readData("datasets/Negative.txt", wsize);
        	calcBPP(datapos, featuresBPP, "datasets/Positive.csv");
        	calcBPP(dataneg, featuresBPP, "datasets/Negative.csv");
		gen = 0;
		cout << "...Done!\n" << endl;
	}else if(gen == 3){
		cout << "Wait... Creating BPP Tables for positive and negative datasets...";
		datapos = readData("datasets/Positive2.txt", wsize);
                dataneg = readData("datasets/Negative2.txt", wsize);
                calcBPP(datapos, featuresBPP, "datasets/Positive2.csv");
                calcBPP(dataneg, featuresBPP, "datasets/Negative2.csv");
		gen = 1;
		cout << "...Done!\n" << endl;
	}
	featureExtractionDNA(data, nthreads, outputfile);
	return 0;
}
