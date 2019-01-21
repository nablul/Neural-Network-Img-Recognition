#ifndef NEURALNETWORK_NEURALNETWORK_H_
#define NEURALNETWORK_NEURALNETWORK_H_

#include <cmath>
#include<fstream>
#include"Matrix.h"

class NeuralNetwork {
private:
int inputSize;
int outputSize;
int numberofHL;

	//Helper function to calculate derivative of sigmoid's function relative to all elements in matrix
	Matrix sigmoid(const Matrix &m);

	//Helper function to turn row from a matrix into a string
	std::string rowToString(Matrix &m, const int &rowNum);

	//Helper function to turn string into a row of a matrix
	void stringToRow(Matrix &m, const std::string str, const int &rowNum);

public:
	std::vector<Matrix> layerMatrices;
	std::vector<Matrix> weightsMatrices;
	std::vector<Matrix> biasMatrices;

	//Default constructor
	NeuralNetwork();

	//Constructor for case when number of hidden layers is provided
	NeuralNetwork(const Matrix &input, const Matrix &output, const std::vector<int> &HLsizes);

	//Constructor for case when number of hidden layers and stored locations for weights and bias matrices are provided
	NeuralNetwork(const Matrix &input, const Matrix &output, const std::vector<int> &HLsizes, const std::string fileWeights, const std::string fileBiases);

	//Calculate each layer of neural network based previous layer and corresponding weights and biases
	void forwardPropagate();

	//Load given matrix as input matrix
	void loadInput(const Matrix &m);

	//Return output matrix
	Matrix returnOutput();

	//Save weights matrices
	void saveWeights(const std::string &filename);

	//Load weights matrices
	void loadWeights(const std::string &filename);

	//Save bias matrices
	void saveBias(const std::string &filename);

	//Load bias matrices
	void loadBias(const std::string &filename);
};

#endif