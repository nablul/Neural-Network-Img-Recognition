#ifndef NEURALNETWORK_NN_TRAINER_H_
#define NEURALNETWORK_NN_TRAINER_H_

#include"NeuralNetwork.h"

class NN_Trainer {
private:
	std::vector<std::vector<std::vector<int>>> RGBInput;
	std::vector<std::vector<double>> trainingOutput;
	std::vector<std::vector<double>> trainingInput;
	std::vector<Matrix> D_weightsMatrices;
	std::vector<Matrix> D_biasMatrices;
	double learningRate;
	int imagewidth = 32;
	int imageheight = 32;

	//Helper function to return a bit at specific position from a byte
	bool getBit(const unsigned char &byte, const int &position);

	//Helper function to turn byte into integer
	int byteToUnsignedInt(const unsigned char &byte);

	//Helper function to calculate derivative of sigmoid's function relative to all elements in matrix
	Matrix sigmoidPrime(const Matrix &m);

	//Populate training input and output vectors with values from specified file
	bool loadTrainingData(const std::string &filename);

	//Return input matrix element for specific training data set
	Matrix loadInputMatrix(const int &dataSet);

	//Return expected output matrix element for specific training data set
	Matrix loadOutputMatrix(const int &dataSet);

	//Calculate D_bias matrices to minimize cost function
	void calculateD_biasMatrices(const Matrix &output, const Matrix &expectedOutput);

	//Calculate D_weights matrices based on bias matrices
	void calculateD_weightsMatrices();

	//Update bias matrices based on D_bias matrices
	void updatebiasMatrices();

	//Update weights matrices based on D_weights matrices
	void updateweightsMatrices();

	//Helper function to calculate the total error (or difference) between two matrices
	double errorPercentCalculator(const Matrix &m1, const Matrix &m2);

	//Helper function that turns anything with 5% mirgin of 0 or 1 into 0 or 1, respectively.
	void stepFunction(Matrix &m);

public:
	NeuralNetwork network;

	//Default constructor
	NN_Trainer();

	//Constructor for case when input, output, neural network layers and learning rate are provided
	NN_Trainer(const int &inputSize, const int &outputSize, const std::vector<int> &HLsizes, const double &learningRate);

	//Run training batch
	void runTrainingBatch(const std::vector<std::string> &trainingBatches);
};

#endif