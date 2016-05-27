#pragma once
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include "NeuralNetworkParameters.h"

using namespace std;

struct SNeuron {
	int numInputs;
	vector <double> vecWeight;
	vector <double> vecInputs;
	double delta;
	SNeuron(int numInputsInput);
};

struct SNeuronLayer {
	int numNeurons;
	vector <SNeuron> vecNeurons;
	SNeuronLayer(int numNeuronsInput, int numInputsPerNeuron);
};

class NeuralNetwork {
public:
	//Constructor
	NeuralNetwork( );
	//Init
	void initializeAgents(NeuralNetworkParameters* parameters);
	//Choosing Actions
	int chooseAction(vector<double>* stateVector);
	vector <double> chooseMultiAction(vector<double>* stateVector);
	//Update
	void update(vector<double> output, vector<double> target);
	//Misc
	vector <double> getOutputs(vector<double> inputs);
	double Sigmoid(double netinput);
	void putWeights(vector<double> weights);
	int GetNumberOfWeights();
	vector <double> getWeights();

	NeuralNetworkParameters* params;
	vector <SNeuronLayer> vecLayers;

	int steps;
	bool lastGreedy;
	vector <double> savedWeights;
	//Destructor
	~NeuralNetwork(void);
};
