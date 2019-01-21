#ifndef NEURALNETWORK_IMAGE_HANDLER_H_
#define NEURALNETWORK_IMAGE_HANDLER_H_

#include <cmath>
#include<assert.h>
#include "bitmap_image.hpp"

class Image_Handler {
private:
	bitmap_image Image;

	//Scan image and return colormap vector
	std::vector<std::vector<int>> colorScanImage(const bitmap_image &image);

	//Create image object from colormap vector and image width and height
	bitmap_image makeImage(const std::vector<std::vector<int>> &colormap, const size_t &width, const size_t &height);

	//Scales bitmap image down to width and height specified. Crops and centeres image if necessary to avoid distortions
	bitmap_image scaledown_Cropcenter(const bitmap_image &original, const int &width, const int &height);

public:
	//Loads bitmap image into colormap vector
	std::vector<std::vector<int>> loadImage(const std::string &filename, const size_t &width, const size_t &height);

	//Print bitmap image given colormap vector, image width and height, and file name
	void printImage(const std::vector<std::vector<int>> &colormap, const size_t &width, const size_t &height, const std::string &filename);
};

#endif