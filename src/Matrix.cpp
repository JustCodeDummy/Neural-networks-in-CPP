//
// Created by firep on 16/06/2026.
//

#include "Matrix.h"

#include <vector>

size_t Matrix::rows() const {
	return this->rows_;
}

size_t Matrix::cols() const {
	return this->cols_;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
	const size_t r_t = this->rows();
	const size_t c_t = this->cols();
	const size_t r_r = rhs.rows();
	const size_t c_r = rhs.cols();

	if (c_t != c_r || r_t != r_r) {
		throw std::runtime_error("Matrix dimension mismatch");
	}
	for (size_t row = 0; row < r_t; row++) {
		for (size_t col = 0; col < c_t; col++) {
			this->data_[row][col] += rhs.data_[row][col];
		}
	}
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& rhs) {
	const size_t r_t = this->rows();
	const size_t c_t = this->cols();
	const size_t r_r = rhs.rows();
	const size_t c_r = rhs.cols();

	if (c_t != c_r || r_t != r_r) {
		throw std::runtime_error("Matrix dimension mismatch");
	}
	for (size_t row = 0; row < r_t; row++) {
		for (size_t col = 0; col < c_t; col++) {
			this->data_[row][col] -= rhs.data_[row][col];
		}
	}
	return *this;
}

Matrix& Matrix::operator*=(float scalar) {
	size_t cols = this->cols();
	size_t rows = this->rows();

	for (size_t row = 0; row < rows; row++) {
		for (size_t col = 0; col < cols; col++) {
			this->data_[row][col] *= scalar;
		}
	}
	return *this;
}

Matrix& Matrix::operator*=(const Matrix& rhs) {
	const size_t r_t = this->rows();
	const size_t c_t = this->cols();
	const size_t r_r = rhs.rows();
	const size_t c_r = rhs.cols();

	if (c_t != r_r) {
		throw std::runtime_error("Matrix dimension mismatch");
	}
	if (c_t<1) {
		throw std::runtime_error("Column count must be a positive integer");
	}
	if (r_t<1) {
		throw std::runtime_error("Row count must be a positive integer");
	}
	std::vector<std::vector<float>> data(r_t, std::vector<float>(c_r, 0.0f));

	for (size_t row = 0; row < r_t; row++) {
		for (size_t col = 0; col < c_t; col++) {
			float sum_k = 0;
			for (size_t k = 0; k < r_r; k++) {
				sum_k += this->data_[row][k] * rhs.data_[k][col];
			}
			data[row][col] = sum_k;
		}
	}
	std::swap(this->data_, data);
	return *this;
}

Matrix operator*(Matrix lhs, const Matrix& rhs) {
	lhs *= rhs;
	return lhs;
}



