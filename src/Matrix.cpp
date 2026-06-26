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
			this->data_[row *r_t+col] += rhs.data_[row*r_t+col];
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
			this->data_[row*c_t + col] -= rhs.data_[row*c_r + col];
		}
	}
	return *this;
}

Matrix& Matrix::operator*=(float scalar) {
	size_t size = cols() * rows();

	for (size_t v = 0; v < size; v++) {
		this->data_[v] *= scalar;
	}
	return *this;
}

float & Matrix::operator()(std::size_t row, std::size_t col) {
	return data_[row * cols_ + col];
}

float Matrix::operator()(std::size_t row, std::size_t col) const {
	return data_[row * cols_ + col];
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
	std::vector<float> data(r_t*c_r, 0.0f);

	for (size_t row = 0; row < r_t; row++) {
		for (size_t col = 0; col < c_r; col++) {
			float sum_k = 0;
			for (size_t k = 0; k < c_t; k++) {
				sum_k += (*this)(row, k) * rhs(k, col);

			}
			 data[row * c_r + col] = sum_k;
		}
	}
	std::swap(this->data_, data);
	return *this;
}

Matrix operator*(Matrix lhs, const Matrix& rhs) {
	lhs *= rhs;
	return lhs;
}



