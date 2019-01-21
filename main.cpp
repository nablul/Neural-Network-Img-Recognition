#include"Image_Handler.h"
#include"NN_Trainer.h"

//Function to process the output matrix and determine the appropriate output label
std::string outputProcessor(const Matrix &output) {
	std::vector<std::string> outputLabels = { "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck" };
	double maxVal = 0;
	int index = 0;

	for (int i = 0; i < output.col; ++i) {
		if (output.array[0][i] > maxVal) {
			maxVal = output.array[0][i];
			index = i;
		};
	};
	return outputLabels[index];
};

//Function to transform image colormap into input matrix for neural network
Matrix colormapToInputMatrix(std::vector<std::vector<int>> colormap) {
	std::vector<double> inputVec(0);

	//Insert each element of colormap into input vector
	for (int i = 0; i < colormap.size(); ++i) {
		for (int j = 0; j < colormap[0].size(); ++j) {
			double tmp = (colormap[i][j] / 255.00000);
			inputVec.push_back(tmp);
		};
	};

	//Create matrix from input vector
	Matrix result(1, inputVec.size());
	for (int i = 0; i < inputVec.size(); ++i) {
		result.array[0][i] = inputVec[i];
	};

	return result;
};

int main() {
	//Initializing neural network parameters
	int inputSize = 3072;
	int outputSize = 10;
	double learningRate = 0.01;
	std::vector<int> HLsizes = { 15 };
	std::vector<std::string> trainingBatches = { "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin" };
	NN_Trainer NN_Session(inputSize, outputSize, HLsizes, learningRate);
	
	//Used to run training batches:
		NN_Session.runTrainingBatch(trainingBatches);

	//Used to classify user specified image:
		////Load exisitng weights and bias files
		//NN_Session.network.loadWeights("Weights.txt");
		//NN_Session.network.loadBias("Bias.txt");

		////Load image from file
		//std::string imageFilePath = "Image5.bmp";
		//Image_Handler imgObj;
		//std::vector<std::vector<int>> colormap = imgObj.loadImage(imageFilePath, 32, 32);
		//imgObj.printImage(colormap, 32, 32, "CroppedCentered" + imageFilePath);

		////Load input data into neural network and forward propagate to retrieve output matrix
		//Matrix inputMatrix = colormapToInputMatrix(colormap);
		//NN_Session.network.loadInput(inputMatrix);
		//NN_Session.network.forwardPropagate();
		//Matrix outputMatrix = NN_Session.network.returnOutput();

		////Process output matrix and display output label
		//std::string outputLabel = outputProcessor(outputMatrix);
		//std::cout << "This is an image of a: " << outputLabel;

	return 0;
};
	