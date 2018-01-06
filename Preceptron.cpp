#include "Perceptron.h"

namespace NN {
	CPerceptron::CPerceptron(int input_size, int vars, const std::vector<int>& layer_sizes, const std::vector<ActivationFunction>& activators, double lr) {
		if (*(layer_sizes.cend() - 1) != 1) {
			throw UnconvergentException();
		}

		int prev_layer = vars;

		for (int i = 0; i < layer_sizes.size(); ++i) {
			this->layers.push_back(std::make_shared<CLayer>(prev_layer, layer_sizes[i], input_size, activators[i]));
			prev_layer = layer_sizes[i];
		}

		this->lr = lr;
	}

	void CPerceptron::fit(const CMatrix &X, const CMatrix &y, int max_iter) {
		// traind model
		for (int i = 0; i < max_iter; ++i) {
			auto& predicted = this->forwardprop(X);
			this->update_grads();
			this->backwardprop(X, y);
			double cost = this->cost(y, predicted);

			if (i % 100 == 0) {
				char outp[100];
				sprintf_s(outp, "iteration: %d, cost: %lf\n", i, cost);
				DBOUT(outp);
			}
		}
	}

	CMatrix CPerceptron::predict(const CMatrix& x) {
		// predict
		auto result = this->forwardprop(x);
		CMatrix matr(result.nrow, result.ncol);

		for (int i = 0; i < result.ncol; ++i) {
			matr[0][i] = 1 ? result[0][i] > .5 : 0;
		}

		return matr;
	}

	CMatrix CPerceptron::forwardprop(const CMatrix& x) {
		// calculate all the coefficients
		auto prev_output = x;

		for (size_t i = 0; i < this->layers.size(); ++i) {
			auto& cur_layer = this->layers[i];

			//cur_layer->Z = cur_layer->W.dot(prev_output) + cur_layer->B;
			cur_layer->Z = cur_layer->W.dot(prev_output);
			for (int i = 0; i < cur_layer->B.ncol; ++i) {
				for (int j = 0; j < cur_layer->Z.nrow; ++j) {
					cur_layer->Z[j][i] = cur_layer->Z[j][i] + cur_layer->B[j][0];
				}
			}


			cur_layer->A = (*cur_layer->activation_func)(cur_layer->Z);

			prev_output = cur_layer->A;
		}

		return (*(this->layers.cend() - 1))->A;
	}

	void CPerceptron::backwardprop(const CMatrix& x, const CMatrix& y) {
		// calculate derivatives

		double m = (double)y.ncol;

		auto& last_layer = *(this->layers.cend() - 1);
		auto& prev_layer = *(this->layers.cend() - 2);

		last_layer->dZ = last_layer->A - y;
		last_layer->dW = last_layer->dZ.dot(prev_layer->A.transpose()) * (1 / m);
		last_layer->dB = last_layer->dZ.sum(1) * (1 / m);
		prev_layer->dA = last_layer->W.transpose().dot(last_layer->dZ);

		for (int i = this->layers.size() - 2; i >= 0; --i) {
			auto& cur_layer = this->layers[i];

			cur_layer->dZ = cur_layer->dA * cur_layer->activation_func->derivative(cur_layer->Z);
			cur_layer->dB = cur_layer->dZ.sum(1) * (1 / m);

			if (i > 0) {
				auto& prev_layer = this->layers[i - 1];

				cur_layer->dW = cur_layer->dZ.dot(prev_layer->A.transpose()) * (1 / m);
				prev_layer->dA = cur_layer->W.transpose().dot(cur_layer->dZ);
			}
			else if (i == 0) {
				cur_layer->dW = cur_layer->dZ.dot(x.transpose()) * (1 / m);
			}
		}
	}

	void CPerceptron::update_grads() {
		// update gradients after forward and backward propagation steps
		for (auto& layer : this->layers) {
			layer->W = layer->W - layer->dW * lr;

			for (int i = 0; i < layer->B.ncol; ++i) {
				for (int j = 0; j < layer->dB.nrow; ++j) {
					layer->B[j][i] = layer->B[j][i] - layer->dB[j][0] * lr;
				}
			}
		}
	}

	double CPerceptron::cost(const CMatrix& real, const CMatrix& nn_out) {
		double total_cost = 0;
		for (int i = 0; i < nn_out.ncol; ++i) {
			double y_real = real[0][i];
			double y_pred = nn_out[0][i];

			// using cross entropy
			double cost = -y_real * log(y_pred);

			total_cost += cost;
		}

		return total_cost;
	}
}