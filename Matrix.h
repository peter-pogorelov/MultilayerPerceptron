#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>

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
		CMatrix(int nrow, int ncol, bool initialize=true);
		CMatrix(const CMatrix& me);

		virtual ~CMatrix();

		void initialize();
		void set_initializer(IInitializer* init);
		double* operator[](int i) const;

		CMatrix dot(const CMatrix & m);
		CMatrix sum(int axis);
		CMatrix operator*(double scalar);
		CMatrix operator*(const CMatrix & m);
		CMatrix operator+(const CMatrix & m);
		CMatrix operator-(const CMatrix & m);
		CMatrix& operator=(const CMatrix & m);

		CMatrix transpose() const;

		std::string to_string();
	private:
		void deep_copy_object(const CMatrix &m);
		void copy_object(const CMatrix &m);
	};
}