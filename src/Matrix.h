//
// Created by firep on 16/06/2026.
//

#ifndef NEURALNETWORKSC_MATRIX_H
#define NEURALNETWORKSC_MATRIX_H


#include <vector>
#include <stdexcept>


// Square matrix
class Matrix  {
	public:
		Matrix(std::size_t rows, std::size_t cols, std::vector<std::vector<float>> data);

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
		std::vector<std::vector<float>> data_;
};


Matrix operator*(Matrix lhs, const Matrix& rhs);




#endif //NEURALNETWORKSC_MATRIX_H