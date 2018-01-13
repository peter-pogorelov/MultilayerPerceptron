#include "MiniBatch.h"

namespace NN {
	void CFullBatch::epoch(std::function<func_iteration> foo) {
		foo(this->x, this->y);
	}

	void CMiniBatch::epoch(std::function<func_iteration> foo) {
		std::default_random_engine generator;
		// make sure that batches are the same within the each epoch
		generator.seed(this->seed);

		std::vector<size_t> indices;
		for (int i = 0; i < this->x.ncol; ++i) {
			indices.push_back(i);
		}

		std::shuffle(
			indices.begin(),
			indices.end(),
			generator
			);

		double ratio = (double)indices.size() / batch_size;
		int minibatches = (int)std::floor(ratio);

		for (int i = 0; i < minibatches; ++i) {
			CMatrix batch_x(this->x.nrow, batch_size);
			CMatrix batch_y(this->y.nrow, batch_size);

			// copying batch column
			for (size_t j = 0; j < (size_t)batch_size; ++j) {
				this->copy_row(batch_x, this->x, j, indices[(i * batch_size) + j]);
				this->copy_row(batch_y, this->y, j, indices[(i * batch_size) + j]);
			}

			foo(batch_x, batch_y);
		}

		if (ratio - minibatches > 0) {
			int shift = batch_size * (minibatches);

			CMatrix batch_x(this->x.nrow, indices.size() - shift);
			CMatrix batch_y(this->y.nrow, indices.size() - shift);

			// copying batch column
			for (size_t j = shift, i = 0; j < indices.size(); ++j, ++i) {
				this->copy_row(batch_x, this->x, i, indices[j]);
				this->copy_row(batch_y, this->y, i, indices[j]);
			}

			foo(batch_x, batch_y);
		}
	}

	void CMiniBatch::copy_row(CMatrix& a, const CMatrix& b, int a_column, int b_column) {
		for (size_t k = 0; k < (size_t)b.nrow; ++k) {
			a[k][a_column] = b[k][b_column];
		}
	}
}