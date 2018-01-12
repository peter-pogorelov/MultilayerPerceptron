#include "Perceptron.h"

namespace NN {
	CPerceptron::CPerceptron(int input_size, int vars, int classes, const std::vector<int>& hidden_layers, const std::vector<ActivationFunction>& activators) {
		int prev_layer = vars;

		for (size_t i = 0; i < hidden_layers.size(); ++i) {
			this->layers.push_back(
				//std::make_shared<CLayer>(prev_layer, hidden_layers[i], input_size, activators[i])
				std::make_shared<CRidgeLayer>(prev_layer, hidden_layers[i], input_size, activators[i], 1)
				//std::make_shared<CDropoutLayer>(prev_layer, hidden_layers[i], input_size, activators[i], 0.1)
			);
			prev_layer = hidden_layers[i];
		}

		int output_neurons = classes == 2 ? 1 : classes;

		this->layers.push_back(
			std::make_shared<CCrossEntropyLayer>(prev_layer, output_neurons, input_size, ActivationFunction::E_SIGMOID)
		);
	}

	void CPerceptron::set_lr(double lr) {
		this->lr = lr;
	}

	void CPerceptron::set_lr_decay_state(bool state) {
		this->lr_decay = state;
	}

	void CPerceptron::fit(const CMatrix &X, const CMatrix &y, int max_iter) {
		// traind model
		double current_lr = this->lr; // make a copy in case of lr decay usage
		
		for (int i = 0; i < max_iter; ++i) {
			auto& predicted = this->forwardprop(X, i);
			this->update_grads();
			this->backwardprop(X, y);
			double cost = this->cost(y, predicted);

			if (this->lr_decay) {
				this->lr = std::pow(current_lr, i);
			}

			if (i % 100 == 0) {
				char outp[100];
				sprintf_s(outp, "epoch: %d, cost: %lf, lr: %lf \n", i, cost, this->lr);
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

	CMatrix CPerceptron::forwardprop(const CMatrix& x, int epoch) {
		// calculate all the coefficients
		auto prev_output = x;

		for (auto& layer : this->layers) {
			layer->forward(prev_output, epoch + time(nullptr) + clock());
			
			prev_output = layer->A;
		}

		return prev_output;
	}

	void CPerceptron::backwardprop(const CMatrix& x, const CMatrix& y) {
		// calculate derivatives
		for (int i = this->layers.size() - 1; i >= 0; --i) {
			auto& cur_layer = this->layers[i];
			this->layers[i]->backward(i > 0 ? this->layers[i-1].get() : nullptr, x, y);
		}
	}

	void CPerceptron::update_grads() {
		// update gradients after forward and backward propagation steps
		for (auto& layer : this->layers) {
			layer->update(this->lr);
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