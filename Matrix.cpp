#include"Matrix.h"

//Default constructor
Matrix::Matrix() {
	row = 0;
	col = 0;
	array = {};
};

//Constructor for specified number of rows & cols
Matrix::Matrix(const int &row, const int &col) {
	this->row = row;
	this->col = col;
	array.resize(row, std::vector<double>(col));
};

//Constructor for creating matrix with another matrix passed as reference
Matrix::Matrix(const std::vector<std::vector<double>> &array) {
	row = array.size();
	col = array[0].size();
	this->array = array;
};

//Scalar multiplication
Matrix Matrix::multiply(const double &val) {
	Matrix result(this->row, this->col);

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			result.array[i][j] = (this->array[i][j])*val;
		};
	};
	return result;
};

//Addition
Matrix Matrix::add(const Matrix &m) {
	//Check for dimensional mismatch
	assert((this->row == m.row) && (this->col == m.col));
	Matrix result(this->row, this->col);

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			result.array[i][j] = this->array[i][j] + m.array[i][j];
		};
	};
	return result;
};

//Subtraction
Matrix Matrix::subtract(const Matrix &m) {
	//Check for dimensional mismatch
	assert((this->row == m.row) && (this->col == m.col));
	Matrix result(this->row, this->col);

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			result.array[i][j] = this->array[i][j] - m.array[i][j];
		};
	};
	return result;
};

//Hadamard product
Matrix Matrix::multiply(const Matrix &m) {
	//Check for dimensional mismatch
	assert((this->row == m.row) && (this->col == m.col));
	Matrix result(this->row, this->col);

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			result.array[i][j] = this->array[i][j] * m.array[i][j];
		};
	};
	return result;
};

//Dot product
Matrix Matrix::dot(const Matrix &m) {
	//Check for dimensional mismatch
	assert(this->col == m.row);
	double tmp = 0;
	Matrix result(this->row, m.col);
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < m.col; ++j) {
			for (int k = 0; k < this->col; ++k) {
				tmp = tmp + this->array[i][k] * m.array[k][j];
			};
			result.array[i][j] = tmp;
			tmp = 0;
		};
	};
	return result;
};

//Transpose
Matrix Matrix::transpose() {
Matrix result(this->col, this->row);
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			result.array[j][i] = this->array[i][j];
		};
	};
	return result;
};

//Populate matrix with random doubles less than 1
void Matrix::populateWithRandom() {
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			this->array[i][j] = (double)((rand() % 10000) + 1) / (10000 + 0.5);
		};
	};
};

//Printing function
void Matrix::printMatrix() {
	for (int i = 0; i < this->row; ++i) {
		for (int j = 0; j < this->col; ++j) {
			std::cout << this->array[i][j] << " ";
		};
		std::cout << "\n";
	};
};
