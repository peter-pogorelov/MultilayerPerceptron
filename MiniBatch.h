#pragma once

#include <functional>
#include <algorithm>
#include <random>

#include "Matrix.h"

namespace NN {
	using func_iteration = void(const CMatrix &x, const CMatrix &y);

	class CTrainMethod {
	protected:
		CMatrix x, y;
		int seed;

	public:
		CTrainMethod(const CMatrix &x, const CMatrix &y, int seed) {
			this->x = x;
			this->y = y;
			this->seed = seed;
		}

		CTrainMethod(int seed) { }

		virtual ~CTrainMethod() { }

		void set_x(const CMatrix& x) {
			this->x = x;
		}

		void set_y(const CMatrix& y) {
			this->y = y;
		}

		virtual void epoch(std::function<func_iteration> foo) = 0;
	};
	
	class CFullBatch : public CTrainMethod {
	public:
		CFullBatch(const CMatrix &x, const CMatrix &y, int seed) :
			CTrainMethod(x, y, seed) {}

		CFullBatch(int seed) : CTrainMethod(seed) { }

		void epoch(std::function<func_iteration> foo);
	};

	class CMiniBatch : public CTrainMethod {
		int batch_size;
	public:
		CMiniBatch(const CMatrix &x, const CMatrix &y, size_t batch_size = 64, int seed = 42) :
			CTrainMethod(x, y, seed)
		{
			this->batch_size = batch_size;
		}

		CMiniBatch(size_t batch_size = 64, int seed = 42) : CTrainMethod(seed) {
			this->batch_size = batch_size;
		}

		void epoch(std::function<func_iteration> foo);

	private:
		void copy_row(CMatrix& a, const CMatrix& b, int a_column, int b_column);
	};
}
