#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include "MatrixInitializer.h"

namespace NN {
	class CMatrix {
	private:
		double** matrix;
		std::shared_ptr<IInitializer> initializer = nullptr;

	public:
		int nrow, ncol;

		class InvalidDimensions {};
		class InvalidInitializer {};

		CMatrix(); // delayed initialization
		CMatrix(int nrow, int ncol, std::shared_ptr<IInitializer> initializer = nullptr);
		CMatrix(const CMatrix& me);

		virtual ~CMatrix();

		void initialize();
		void set_initializer(IInitializer* init);
		double* operator[](int i) const;

		CMatrix dot(const CMatrix & m);
		CMatrix sum(int axis);
		CMatrix add_vector(const CMatrix& m);
		CMatrix multiply_vector(const CMatrix& m);
		CMatrix operator*(double scalar);
		CMatrix operator*(const CMatrix & m);
		CMatrix operator+(const CMatrix & m);
		CMatrix operator-(const CMatrix & m);
		CMatrix operator-();
		CMatrix& operator=(const CMatrix & m);

		CMatrix transpose() const;
		CMatrix shuffle(int seed = 42) const;
		CMatrix vertical_slice(int from, int to) const;
		CMatrix horizontal_slize(int from, int to) const;

		double** get();

		std::string to_string();
	private:
		void deep_copy_object(const CMatrix &m);
		void copy_object(const CMatrix &m);
	};
}