#include"NeuralNetwork.h"

//Helper function to calculate derivative of sigmoid's function relative to all elements in matrix
Matrix NeuralNetwork::sigmoid(const Matrix &m) {
	Matrix result(m.row, m.col);
	for (int i = 0; i < m.row; ++i) {
		for (int j = 0; j < m.col; ++j) {
			double tmp = m.array[i][j];
			//For each element, apply sigmoid's function
			result.array[i][j] = 1 / (1 + exp(-(tmp)));
		};
	};
	return result;
};

//Helper function to turn row from a matrix into a string
std::string NeuralNetwork::rowToString(Matrix &m, const int &rowNum) {
	std::string result = "";

	for (int i = 0; i < m.col; ++i) {
		result = result + std::to_string(m.array[rowNum][i]) + "|";
	};
	return result;
};

//Helper function to turn string into a row of a matrix
void NeuralNetwork::stringToRow(Matrix &m, const std::string str, const int &rowNum) {
	std::string inputStr = str;
	std::string delimiter = "|";
	int pos = 0;
	std::string tmp;
	int colCounter = 0;

	//For each delimiter, convert accompanying data into double and store into matrix
	while (((pos = inputStr.find(delimiter)) != std::string::npos) && ((pos = inputStr.find(delimiter)) > 0)) {
		tmp = inputStr.substr(0, pos);
		m.array[rowNum][colCounter] = stod(tmp);
		inputStr.erase(0, pos + delimiter.length());
		++colCounter;
	};
};

//Default constructor
NeuralNetwork::NeuralNetwork() {
	inputSize = 5;
	outputSize = 5;
	numberofHL = 2;
	int typicalHLsize = 10;

	//Calculate neural network layer matrices
	layerMatrices.resize(numberofHL + 2);
	layerMatrices[0] = Matrix(1, inputSize);
	for (int i = 1; i < numberofHL + 1; ++i) {
		layerMatrices[i] = Matrix(1, typicalHLsize);
	};
	layerMatrices[numberofHL + 1] = Matrix(1, outputSize);

	//Create weight matrices
	weightsMatrices.resize(numberofHL + 1);
	for (int i = 0; i < numberofHL + 1; ++i) {
		weightsMatrices[i] = Matrix(layerMatrices[i].col, layerMatrices[i + 1].col);
		//Populate with random doubles
		weightsMatrices[i].populateWithRandom();
	};

	//Create bias matrices
	biasMatrices.resize(numberofHL + 1);
	for (int i = 0; i < numberofHL + 1; ++i) {
		biasMatrices[i] = Matrix(1, layerMatrices[i + 1].col);
		//Populate with random doubles
		biasMatrices[i].populateWithRandom();
	};
};

//Constructor for case when number of hidden layers is provided
NeuralNetwork::NeuralNetwork(const Matrix &input, const Matrix &output, const std::vector<int> &HLsizes) {
	numberofHL = HLsizes.size();

	//Calculate neural network layer matrices
	layerMatrices.resize(numberofHL + 2);
	layerMatrices[0] = input;
	int HLsizeCounter = 0;
	for (int i = 1; i < numberofHL + 1; ++i) {
		layerMatrices[i] = Matrix(1, HLsizes[HLsizeCounter]);
		++HLsizeCounter;
	};
	layerMatrices[numberofHL + 1] = output;

	//Create weight matrices
	weightsMatrices.resize(numberofHL + 1);
	for (int i = 0; i < numberofHL + 1; ++i) {
		weightsMatrices[i] = Matrix(layerMatrices[i].col, layerMatrices[i + 1].col);
		//Populate with random doubles
		weightsMatrices[i].populateWithRandom();
	};

	//Create bias matrices
	biasMatrices.resize(numberofHL + 1);
	for (int i = 0; i < numberofHL + 1; ++i) {
		biasMatrices[i] = Matrix(1, layerMatrices[i + 1].col);
		//Populate with random doubles
		biasMatrices[i].populateWithRandom();
	};
};

//Constructor for case when number of hidden layers and stored locations for weights and bias matrices are provided
NeuralNetwork::NeuralNetwork(const Matrix &input, const Matrix &output, const std::vector<int> &HLsizes, const std::string fileWeights, const std::string fileBiases) {
	numberofHL = HLsizes.size();

	//Calculate neural network layer matrices
	layerMatrices.resize(numberofHL + 2);
	layerMatrices[0] = input;
	int HLsizeCounter = 0;
	for (int i = 1; i < numberofHL + 1; ++i) {
		layerMatrices[i] = Matrix(1, HLsizes[HLsizeCounter]);
		++HLsizeCounter;
	};
	layerMatrices[numberofHL + 1] = output;

	//Create weight matrices
	weightsMatrices.resize(numberofHL + 1);
	for (int i = 0; i < numberofHL + 1; ++i) {
		weightsMatrices[i] = Matrix(layerMatrices[i].col, layerMatrices[i + 1].col);
	};

	//Create bias matrices
	biasMatrices.resize(numberofHL + 1);
	for (int i = 0; i < numberofHL + 1; ++i) {
		biasMatrices[i] = Matrix(1, layerMatrices[i + 1].col);
	};

	//Load weights and bias matrices
	loadWeights(fileWeights);
	loadBias(fileBiases);
};

//Calculate each layer of neural network based previous layer and corresponding weights and biases
void NeuralNetwork::forwardPropagate() {
	for (int i = 1; i < layerMatrices.size(); ++i) {
		layerMatrices[i] = sigmoid(layerMatrices[i - 1].dot(weightsMatrices[i - 1]).add(biasMatrices[i - 1]));
	};
};

//Load given matrix as input matrix
void NeuralNetwork::loadInput(const Matrix &m) {
	layerMatrices[0] = Matrix(m);
};

//Return output matrix
Matrix NeuralNetwork::returnOutput() {
	return layerMatrices[layerMatrices.size() - 1];
};

//Save weights matrices
void NeuralNetwork::saveWeights(const std::string &filename) {
	std::ofstream outfile;
	outfile.open(filename, std::ofstream::out);
	std::string tmp;

	//Convert matrix row into string and store to file
	for (int i = 0; i < weightsMatrices.size(); ++i) {
		for (int j = 0; j < weightsMatrices[i].row; ++j) {
			tmp = rowToString(weightsMatrices[i], j);
			outfile << tmp << "\n";
		};
	};
	outfile.close();
	return;
};

//Load weights matrices
void NeuralNetwork::loadWeights(const std::string &filename) {
	std::ifstream infile;
	infile.open(filename, std::ofstream::out);
	std::string fileline;
	int matrixCounter = 0;
	int rowCounter = -1;

	//While reading file, convert and save string from file into matrix row
	while (std::getline(infile, fileline)) {
		if (rowCounter < weightsMatrices[matrixCounter].row) {
			++rowCounter;
		};
		stringToRow(weightsMatrices[matrixCounter], fileline, rowCounter);
		if (rowCounter >= weightsMatrices[matrixCounter].row - 1) {
			++matrixCounter;
			rowCounter = -1;
		};
	};
	infile.close();
	return;
};

//Save bias matrices
void NeuralNetwork::saveBias(const std::string &filename) {
	std::ofstream outfile;
	outfile.open(filename, std::ofstream::out);
	std::string tmp;

	//Convert matrix row into string and store to file
	for (int i = 0; i < biasMatrices.size(); ++i) {
		for (int j = 0; j < biasMatrices[i].row; ++j) {
			tmp = rowToString(biasMatrices[i], j);
			outfile << tmp << "\n";
		};
	};
	outfile.close();
	return;
};

//Load bias matrices
void NeuralNetwork::loadBias(const std::string &filename) {
	std::ifstream infile;
	infile.open(filename, std::ofstream::out);
	std::string fileline;
	int matrixCounter = 0;
	int rowCounter = -1;

	//While reading file, convert and save string from file into matrix row
	while (std::getline(infile, fileline)) {
		if (rowCounter < biasMatrices[matrixCounter].row) {
			++rowCounter;
		};
		stringToRow(biasMatrices[matrixCounter], fileline, rowCounter);
		if (rowCounter >= biasMatrices[matrixCounter].row - 1) {
			++matrixCounter;
			rowCounter = -1;
		};
	};
	infile.close();
	return;
};
