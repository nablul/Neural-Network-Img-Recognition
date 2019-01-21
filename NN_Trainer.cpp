#include"NN_Trainer.h"

//Helper function to return a bit at specific position from a byte
bool NN_Trainer::getBit(const unsigned char &byte, const int &position) {
	return (byte >> position) & 0x1;
};

//Helper function to turn byte into integer
int NN_Trainer::byteToUnsignedInt(const unsigned char &byte) {
	int result = 0;

	for (int i = 0; i < 8; ++i) {
		result = result + (getBit(byte, i) * pow(2, i));
	};

	return result;
};

//Helper function to calculate derivative of sigmoid's function relative to all elements in matrix
Matrix NN_Trainer::sigmoidPrime(const Matrix &m) {
	Matrix result(m.row, m.col);
	for (int i = 0; i < m.row; ++i) {
		for (int j = 0; j < m.col; ++j) {
			double tmp = m.array[i][j];
			//For each element, apply derivative of sigmoid's function
			result.array[i][j] = exp(-tmp) / (pow(1 + exp(-tmp), 2));
		};
	};
	return result;
};

//Populate training input and output vectors with values from specified file
bool NN_Trainer::loadTrainingData(const std::string &filename) {

	std::ifstream inFile(filename, std::ios::out | std::ios::binary);

	if (inFile) {

		//Calculate length of file
		inFile.seekg(0, inFile.end);
		int filelength = inFile.tellg();
		inFile.seekg(0, inFile.beg);

		//Buffer to store input data
		char *buffer = new char[filelength];

		//Read full file into buffer
		inFile.read(buffer, filelength);

		//Check file not closed while reading
		if (!inFile) {
			return false;
		};

		//Close file
		inFile.close();

		//Calculate size of training set and number of elements in training set
		int dataSetSize = ((imagewidth * imageheight * 3) + 1);
		int numOfDataSets = filelength / dataSetSize;

		//Populate training input and output matrices
		trainingInput.resize(numOfDataSets, std::vector<double>(0));
		trainingOutput.resize(numOfDataSets, std::vector<double>(10));
		RGBInput.resize(numOfDataSets, std::vector<std::vector<int>>(3, std::vector<int>(0)));

		//Load RGB, input and output values for each data set
		for (int i = 0; i < numOfDataSets; ++i) {

			//Load correct training output values
			int expectedOutput = byteToUnsignedInt(buffer[i*dataSetSize]);
			trainingOutput[i][expectedOutput] = 1.0;

			//Load j[0] = R pixel, j[1] = G pixel, j[2] = B pixel into RGBinput
			for (int j = 0; j < 3; ++j) {
				for (int k = 0; k < imagewidth*imageheight; ++k) {
					RGBInput[i][j].push_back(byteToUnsignedInt(buffer[(1 + (i * dataSetSize) + (j*imagewidth*imageheight) + k)]));

					//Populate training input values
					double tmp = (RGBInput[i][j][k] / 255.00000);
					trainingInput[i].push_back(tmp);
				};
			};
		};
		delete buffer;
		return true;
	}
	else {
		return false;
	};
};

//Return input matrix element for specific training data set
Matrix NN_Trainer::loadInputMatrix(const int &dataSet) {
	//Calculate size of input for given dataset number
	int numberOfInputs = trainingInput[dataSet].size();
	std::vector<std::vector<double>> inputArray(1, std::vector<double>(0));

	//Load input data
	for (int i = 0; i < numberOfInputs; ++i) {
		double value = (double)trainingInput[dataSet][i];
		inputArray[0].push_back(value);
	};
	//Create new matrix with loaded data and return
	return Matrix(inputArray);
};

//Return expected output matrix element for specific training data set
Matrix NN_Trainer::loadOutputMatrix(const int &dataSet) {
	//Calculate size of output for given dataset number
	int numberOfOutputs = trainingOutput[dataSet].size();
	std::vector<std::vector<double>> outputArray(1, std::vector<double>(0));

	//Load output data
	for (int i = 0; i < numberOfOutputs; ++i) {
		double value = (double)trainingOutput[dataSet][i];
		outputArray[0].push_back(value);
	};
	//Create new matrix with loaded data and return
	return Matrix(outputArray);
};

//Calculate D_bias matrices to minimize cost function
void NN_Trainer::calculateD_biasMatrices(const Matrix &output, const Matrix &expectedOutput) {
	Matrix outputCopy(output);
	for (int i = D_biasMatrices.size() - 1; i >= 0; --i) {
		if (i == D_biasMatrices.size() - 1) {
			D_biasMatrices[i] = (outputCopy.subtract(expectedOutput)).multiply(sigmoidPrime((network.layerMatrices[i].dot(network.weightsMatrices[i])).add(network.biasMatrices[i])));
		}
		else {
			(D_biasMatrices[i] = D_biasMatrices[i + 1].dot(network.weightsMatrices[i+1].transpose())).multiply(sigmoidPrime((network.layerMatrices[i].dot(network.weightsMatrices[i])).add(network.biasMatrices[i])));
		};
	};
};

//Calculate D_weights matrices based on bias matrices
void NN_Trainer::calculateD_weightsMatrices() {
	for (int i = 0; i < D_weightsMatrices.size(); ++i) {
		D_weightsMatrices[i] = (network.layerMatrices[i].transpose()).dot(D_biasMatrices[i]);
	};
};

//Update bias matrices based on D_bias matrices
void NN_Trainer::updatebiasMatrices() {
	for (int i = 0; i < D_biasMatrices.size(); ++i) {
		network.biasMatrices[i] = network.biasMatrices[i].subtract(D_biasMatrices[i].multiply(learningRate));
	};
};

//Update weights matrices based on D_weights matrices
void NN_Trainer::updateweightsMatrices() {
	for (int i = 0; i < D_weightsMatrices.size(); ++i) {
		network.weightsMatrices[i] = network.weightsMatrices[i].subtract(D_weightsMatrices[i].multiply(learningRate));
	};
};

//Helper function to calculate the total error (or difference) between two matrices
double NN_Trainer::errorPercentCalculator(const Matrix &m1, const Matrix &m2) {
	assert(m1.row == m2.row);
	assert(m1.col == m2.col);
	double errorCounter = 0;

	for (int i = 0; i < m1.row; ++i) {
		for (int j = 0; j < m1.col; ++j) {
			errorCounter += abs(m1.array[i][j] - m2.array[i][j]);
		};
	};
	return (errorCounter)/(m1.row*m1.col);
};

//Helper function that turns anything with 5% mirgin of 0 or 1 into 0 or 1, respectively.
void NN_Trainer::stepFunction(Matrix &m) {
	for (int i = 0; i < m.row; ++i) {
		for (int j = 0; j < m.col; ++j) {
			if (m.array[i][j] < 0.1) {
				m.array[i][j] = 0;
			};
			if (m.array[i][j] > 0.9) {
				m.array[i][j] = 1;
			};
		};
	};
};

//Default constructor
NN_Trainer::NN_Trainer() {};

//Constructor for case when input, output, neural network layers and learning rate are provided
NN_Trainer::NN_Trainer(const int &inputSize, const int &outputSize, const std::vector<int> &HLsizes, const double &learningRate) {
	this->learningRate = learningRate;
	Matrix inputMatrix(1, inputSize);
	Matrix outputMatrix(1, outputSize);
	network = NeuralNetwork(inputMatrix, outputMatrix, HLsizes);

	//Calculate D_bias matrices
	D_biasMatrices.resize(network.biasMatrices.size());
	for (int i = 0; i < network.biasMatrices.size(); ++i) {
		D_biasMatrices[i] = Matrix(network.biasMatrices[i].row, network.biasMatrices[i].col);
	};

	//Calculate D_weights matrices
	D_weightsMatrices.resize(network.weightsMatrices.size());
	for (int i = 0; i < network.weightsMatrices.size(); ++i) {
		D_weightsMatrices[i] = Matrix(network.weightsMatrices[i].row, network.weightsMatrices[i].col);
	};
};

//Run training batch
void NN_Trainer::runTrainingBatch(const std::vector<std::string> &trainingBatches) {
	Matrix output;
	Matrix expectedOutput;
	std::vector<double> errorPercent;
	double averagePercentError = 0;
	int numberofDataSets;
	int runNumber = 1;

	//For each training batch
	for (int i = 0; i < trainingBatches.size(); ++i) {
		loadTrainingData(trainingBatches[i]);
		numberofDataSets = trainingInput.size();

		//Load bias and weights matrices if they are available from previous training batches
		if (runNumber > 1) {
			network.loadBias("Bias.txt");
			network.loadWeights("Weights.txt");
		};

		//For each data set
		for (int j = 0; j < numberofDataSets; ++j) {
			network.loadInput(loadInputMatrix(j));

			//Forward propagate network and retrive output
			network.forwardPropagate();
			expectedOutput = loadOutputMatrix(j);
			output = network.returnOutput();

			// Calculate cost function gradient and back propagate network. Update weights and bias matrices
			calculateD_biasMatrices(output, expectedOutput);
			calculateD_weightsMatrices();
			updatebiasMatrices();
			updateweightsMatrices();

			//Calculate and save error for data set
			errorPercent.push_back(errorPercentCalculator(output, expectedOutput));

			//Progress update
			std::cout << "Dataset " << j << " for training batch " << i << " Processed.\n";
			std::cout << "Error: " << errorPercentCalculator(output, expectedOutput) << "\n\n";
		};

		//Update bias and weights files
		std::ofstream biasFile;
		std::ofstream weightsFile;
		biasFile.open("Bias.txt", std::ofstream::out | std::ofstream::trunc);
		biasFile.close();
		weightsFile.open("Weights.txt", std::ofstream::out | std::ofstream::trunc);
		weightsFile.close();
		network.saveBias("Bias.txt");
		network.saveWeights("Weights.txt");

		//Calculate average error
		for (int k = 0; k < numberofDataSets; ++k) {
			averagePercentError += errorPercent[k];
		};

		//Display average error for training batch
		averagePercentError = averagePercentError / (errorPercent.size());
		std::cout << "Average error % for training batch: " << i << averagePercentError*100 << "\n";

		averagePercentError = 0;
		errorPercent.resize(0);
		++runNumber;
	};
};