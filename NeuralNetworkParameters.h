#pragma once
#include <vector>
#include <cstdlib>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <string.h>
#include <iostream>

using namespace std;
struct NeuralNetworkParameters {
	int numInputs;
	int numOutputs;
	int numHiddenLayers;
	int neuronsPerHiddenLayer;
	double maxVal, minVal;
};

/*
inline void NeuralNetworkParameters::parseParameters(string* input) {
	numInputs = 1;
	numOutputs = 2;
	numHiddenLayers = 2;
	neuronsPerHiddenLayer = 10;

	maxVal = 1;
	minVal = 0;

	//Files are in the form of "SimulatorName-SimulatorParseValue.txt" This can be found in SimulatorCreator.cpp.
	ifstream data(input->c_str());


	if(!data.is_open()) {
		printf("File %s not found. Using default arguments\n", input->c_str());
	}
	string line;

	while(std::getline(data,line,'=')) {
		if (FillVariableParsing::fillVariable(&data, line,numInputs,"numInputs"));
		else if (FillVariableParsing::fillVariable(&data, line,numOutputs,"numOutputs"));
		else if (FillVariableParsing::fillVariable(&data, line,numHiddenLayers,"numHiddenLayers"));
		else if (FillVariableParsing::fillVariable(&data, line,neuronsPerHiddenLayer,"neuronsPerHiddenLayer"));
		else getline(data, line);
	}
}
*/