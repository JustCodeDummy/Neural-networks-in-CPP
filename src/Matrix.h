#pragma once
//
// Created by firep on 16/06/2026.
//


#include <vector>
#include <stdexcept>


// Square matrix
class Matrix  {
	public:
		Matrix(std::size_t rows, std::size_t cols, std::vector<float> data)
			:rows_(rows), cols_(cols), data_(std::move(data)){
			if (rows == 0 || cols == 0) {
				throw std::runtime_error("Matrix dimensions must be positive");
			}
			if (data_.size() != rows_ * cols_) {
				throw std::runtime_error("Matrix data size does not match dimensions");
			}
		}

		float& operator()(std::size_t row, std::size_t col);
		float operator()(std::size_t row, std::size_t col) const;
		Matrix& operator*=(const Matrix& rhs);
		Matrix& operator+=(const Matrix& other);
		Matrix& operator-=(const Matrix& rhs);
		Matrix& operator*=(float scalar);

		[[nodiscard]] std::size_t rows() const;
		[[nodiscard]] std::size_t cols() const;

	private:
		std::size_t rows_;
		std::size_t cols_;
		std::vector<float> data_;
};


Matrix operator*(Matrix lhs, const Matrix& rhs);




