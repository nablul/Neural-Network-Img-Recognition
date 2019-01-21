#ifndef NEURALNETWORK_MATRIX_H_
#define NEURALNETWORK_MATRIX_H_

#include<assert.h>
#include<iostream>
#include"vector"
#include"string"

struct Matrix {
	int row;
	int col;
	std::vector <std::vector<double>> array;

	//Default constructor
	Matrix();

	//Constructor for specified number of rows & cols
	Matrix(const int &row, const int &col);

	//Constructor for creating matrix with another matrix passed as reference
	Matrix(const std::vector<std::vector<double>> &array);

	//Scalar multiplication
	Matrix multiply(const double &val);

	//Addition
	Matrix add(const Matrix &m);

	//Subtraction
	Matrix subtract(const Matrix &m);

	//Hadamard product
	Matrix multiply(const Matrix &m);

	//Dot product
	Matrix dot(const Matrix &m);

	//Transpose
	Matrix transpose();

	//Populate matrix with random doubles less than 1
	void populateWithRandom();

	//Printing function
	void printMatrix();
};

#endif