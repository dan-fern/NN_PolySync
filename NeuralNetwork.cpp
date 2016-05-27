#include "NeuralNetwork.h"

SNeuron::SNeuron(int numInputsInput) {
	numInputs = numInputsInput + 1;
	for(int i = 0; i < numInputs; i++) {
		vecWeight.push_back(rand()/double(RAND_MAX));
		vecInputs.push_back(0);
	}
}

SNeuronLayer::SNeuronLayer(int numNeuronsInput, int numInputsPerNeuron) {
	numNeurons = numNeuronsInput;
	for (int i=0; i<numNeurons; ++i) vecNeurons.push_back(SNeuron(numInputsPerNeuron));
}

NeuralNetwork::NeuralNetwork() {

}

void NeuralNetwork::initializeAgents(NeuralNetworkParameters* parametersInput) {
	steps = 0;
	lastGreedy = false;
	params = parametersInput;	
	//create the layers of the network
	if (params->numHiddenLayers > 0) {
		//create first hidden layer
		vecLayers.push_back(SNeuronLayer(params->neuronsPerHiddenLayer, params->numInputs));
		for (int i=0; i<params->numHiddenLayers-1; ++i)	{
				vecLayers.push_back(SNeuronLayer(params->neuronsPerHiddenLayer, params->neuronsPerHiddenLayer));
		}
		//create output layer
		vecLayers.push_back(SNeuronLayer(params->numOutputs, params->neuronsPerHiddenLayer));
	} else {
	  //create output layer
	  vecLayers.push_back(SNeuronLayer(params->numOutputs, params->numInputs));
  }
}

void NeuralNetwork::update(vector<double> output, vector<double> target) {
	//Compute deltas
	for(int i = params->numHiddenLayers; i >= 0; i--) {
		for(int j = 0; j < vecLayers[i].numNeurons; j++) {
			if(i == params->numHiddenLayers) {
				double delta = (output[j] - target[j]) * output[j] * (1 - output[j]);
				vecLayers[i].vecNeurons[j].delta = delta;
			} else {
				double sum = 0;
				for(int j2 = 0; j2 < vecLayers[i+1].numNeurons; j2++) {
					double weight = vecLayers[i+1].vecNeurons[j2].vecWeight[j];
					double delta = vecLayers[i+1].vecNeurons[j2].delta;
					double inputs = vecLayers[i+1].vecNeurons[j2].vecInputs[j];
					sum += weight * delta * inputs * (1 - inputs);
				}
				double delta = sum;
				vecLayers[i].vecNeurons[j].delta = delta;			
			}
		}
	}
	for(int i = params->numHiddenLayers; i >= 0 ; i--) {
		for(int j = 0; j < vecLayers[i].numNeurons; j++) {
			for(int k = 0; k < vecLayers[i].vecNeurons[j].numInputs; k++) {
				double* weight = &vecLayers[i].vecNeurons[j].vecWeight[k];
				*weight -= 0.4 * vecLayers[i].vecNeurons[j].delta * vecLayers[i].vecNeurons[j].vecInputs[k];
			}
		}
	}
}

int NeuralNetwork::chooseAction(vector<double>* stateVector) {
	vector <double> outputs = getOutputs(*stateVector);
	int action = int(outputs[0]);
	return action;
}

vector <double> NeuralNetwork::chooseMultiAction(vector<double>* stateVector) {
	vector<double> outputs = getOutputs(*stateVector);
	return outputs;
}
vector <double> NeuralNetwork::getOutputs(vector<double> inputs) {
	vector <double> outputs;
	int cWeight = 0;
	if(inputs.size() != params->numInputs) {
			printf("NN: input size() != numInputs\n");
			cin.get();
			exit(1);
	}
	for(int i = 0; i < params->numHiddenLayers +1; i++) {
		//If first iteration, sigmoidify (activate) the inputs. 
		if (i == 0) {
			for(int j = 0; j < inputs.size(); j++) {
				inputs[j] = Sigmoid(inputs[j]);
				//Sigmoid goes here. I think there is a NN library that I could have called
			}
		}
		if (i > 0) {
			inputs = outputs;
			outputs.clear();
		}
		outputs.clear();
		cWeight = 0;
		for(int j = 0; j < vecLayers[i].numNeurons; j++) {
			double netinput = 0;
			int numInputs = vecLayers[i].vecNeurons[j].numInputs;
			//Loop through each weight
			for(int k = 0; k < numInputs - 1; k++) {
				//Sum weights for inputs
				vecLayers[i].vecNeurons[j].vecInputs[k] = inputs[cWeight];
				netinput += vecLayers[i].vecNeurons[j].vecWeight[k] * inputs[cWeight++];
			}
			//Add bias
			netinput += vecLayers[i].vecNeurons[j].vecWeight[numInputs-1] * -1.0;
			outputs.push_back(Sigmoid(netinput));
			//Or equivalent backpropagation
			cWeight = 0;
		}
	}

	//for(int i = 0; i < outputs.size(); i++) {
		//outputs[i] = (params->maxVal + abs(params->minVal) ) * (outputs[i]) - abs(params->minVal);
		
		/*if(outputs[i] >= 0.9) 
			outputs[i] = params->maxVal;
		else if(outputs[i] <= 0.1) 
			outputs[i] = params->minVal;
		else {
			//outputs[i] = (outputs[i] - 0.1 + params->minVal)/(params->minVal + params->maxVal) * params->maxVal;
			outputs[i] = (outputs[i] - 0.1)/(0.8) * (params->maxVal - params->minVal) + params->minVal;
		}
		*/
	//}
	return outputs;
}

double NeuralNetwork::Sigmoid(double netinput) {
	/* 
	static map <int, double> sigmoidMap;
	int tempInput = netinput * 10;
	map <int, double>::iterator it = sigmoidMap.find(tempInput);

	if(it != sigmoidMap.end()) {
		return it->second;
	} else {
		double val =  ( 1 / ( 1 + exp(-netinput/1.0)));
		sigmoidMap.insert(make_pair(tempInput, val));
		return val;
	}
	*/
	return  ( 1 / ( 1 + exp(-netinput/1.0)));
	//subject to change
}

void NeuralNetwork::putWeights(vector<double> weights) {
	int cWeight = 0;
	//for each layer
	for (int i=0; i<params->numHiddenLayers + 1; ++i) {
		//for each neuron
		for (int j=0; j<vecLayers[i].numNeurons; ++j) {
			//for each weight
			for (int k=0; k<vecLayers[i].vecNeurons[j].numInputs; ++k) {
				vecLayers[i].vecNeurons[j].vecWeight[k] = weights[cWeight++];
			}
		}
	}
	return;
}

vector <double> NeuralNetwork::getWeights() {
	int cWeight = 0;
	vector<double> weights;
	//for each layer
	for (int i=0; i<params->numHiddenLayers + 1; ++i) {
		//for each neuron
		for (int j=0; j<vecLayers[i].numNeurons; ++j) {
			//for each weight
			for (int k=0; k<vecLayers[i].vecNeurons[j].numInputs; ++k) {
				weights.push_back(vecLayers[i].vecNeurons[j].vecWeight[k]);
			}
		}
	}
	return weights;
}

int NeuralNetwork::GetNumberOfWeights() {
	int weights = 0;
	//for each layer
	for (int i=0; i<params->numHiddenLayers + 1; ++i) {
		//for each neuron
		for (int j=0; j<vecLayers[i].numNeurons; ++j) {
			//for each weight
			for (int k=0; k<vecLayers[i].vecNeurons[j].numInputs; ++k) {
				weights++;
			}
		}
	}
	return weights;
}

NeuralNetwork::~NeuralNetwork(void)
{
}