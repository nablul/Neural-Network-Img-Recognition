#include "Image_Handler.h"

//Scan image and return colormap vector
std::vector<std::vector<int>> Image_Handler::colorScanImage(const bitmap_image &image) {
	std::vector<std::vector<int>> colormap(3, std::vector<int>(0));
	size_t height = image.height();
	size_t width = image.width();

	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {

			//Populate RGB vectors
			rgb_t color;
			image.get_pixel(x, y, color);
			colormap[0].push_back(color.red);
			colormap[1].push_back(color.green);
			colormap[2].push_back(color.blue);
		};
	};
	return colormap;
};

//Create image object from colormap vector and image width and height
bitmap_image Image_Handler::makeImage(const std::vector<std::vector<int>> &colormap, const size_t &width, const size_t &height) {
	bitmap_image image(width, height);
	int colormapCounter = 0;

	//Populate RGB values
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			rgb_t color;
			color.red = colormap[0][colormapCounter];
			color.green = colormap[1][colormapCounter];
			color.blue = colormap[2][colormapCounter];
			++colormapCounter;
			image.set_pixel(x, y, color);
		};
	};
	return image;
};

//Scales bitmap image down to width and height specified. Crops and centeres image if necessary to avoid distortions
bitmap_image Image_Handler::scaledown_Cropcenter(const bitmap_image &original, const int &width, const int &height) {
	assert(width <= original.width());
	assert(height <= original.height());

	int widthScaleFactor = 0;
	int heightScaleFactor = 0;
	int minScaleFactor;

	//Calculate scale required for width dimension
	while (width*pow(2, widthScaleFactor + 1) <= original.width()) {
		++widthScaleFactor;
	};

	//Calculate scale required for height dimension
	while (height*pow(2, heightScaleFactor + 1) <= original.height()) {
		++heightScaleFactor;
	};

	//Calculate governing scale
	if (widthScaleFactor <= heightScaleFactor) {
		minScaleFactor = widthScaleFactor;
	}
	else {
		minScaleFactor = heightScaleFactor;
	}

	//Crop image size
	size_t cropImageWidth = width * pow(2, minScaleFactor);
	size_t cropImageHeight = height * pow(2, minScaleFactor);

	int bottomLeftCorner_X = (int)(original.width() - cropImageWidth)*0.5;
	int bottomLeftCorner_Y = (int)(original.height() - cropImageHeight)*0.5;

	//Copy cropped image onto new image object
	bitmap_image result(cropImageWidth, cropImageHeight);
	for (size_t y = 0; y < cropImageHeight; ++y) {
		for (size_t x = 0; x < cropImageWidth; ++x) {
			rgb_t color;
			original.get_pixel(x + bottomLeftCorner_X, y + bottomLeftCorner_Y, color);
			result.set_pixel(x, y, color);
		};
	};

	//Scale image with sub_scaling function
	std::vector<bitmap_image> imgScalingVec;
	imgScalingVec.resize(minScaleFactor);
	if (minScaleFactor == 1) {
		result.subsample(imgScalingVec[0]);
		return result;
	}
	else {
		result.subsample(imgScalingVec[0]);
		for (int i = 1; i < minScaleFactor; ++i) {
			imgScalingVec[i - 1].subsample(imgScalingVec[i]);
		};
		return imgScalingVec[imgScalingVec.size() - 1];
	};
};

//Loads bitmap image into colormap vector
std::vector<std::vector<int>> Image_Handler::loadImage(const std::string &filename, const size_t &width, const size_t &height) {
	bitmap_image image(filename);

	if (!image) {
		std::vector<std::vector<int>> colormap(0, std::vector<int>(0));
		return colormap;
	}
	else {
		bitmap_image scaledCroppedImg = scaledown_Cropcenter(image, width, height);
		return colorScanImage(scaledCroppedImg);
	};
};

//Print bitmap image given colormap vector, image width and height, and file name
void Image_Handler::printImage(const std::vector<std::vector<int>> &colormap, const size_t &width, const size_t &height, const std::string &filename) {
	bitmap_image image = makeImage(colormap, width, height);
	image.save_image(filename);
	return;
};
