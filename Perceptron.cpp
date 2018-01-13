#include "Perceptron.h"

namespace NN {
	CPerceptron::CPerceptron(
		int input_size,
		int vars,
		int classes,
		std::vector<std::tuple<layer_size, ActivationFunction, LayerTypes>>& layer_params
	) {

		int prev_layer = vars;

		for (auto& param : layer_params) {
			layer_size l_size = std::get<0>(param);
			ActivationFunction l_activation = std::get<1>(param);
			LayerTypes l_type = std::get<2>(param);

			switch (l_type) {
			case LayerTypes::E_PLAIN:
				this->layers.push_back(std::make_shared<CLayer>(prev_layer, l_size, input_size, l_activation));
				break;
			case LayerTypes::E_RIDGE:
				// at the moment the penalty of ridge layer is fixed to one
				this->layers.push_back(std::make_shared<CRidgeLayer>(prev_layer, l_size, input_size, l_activation, 1));
				break;
			case LayerTypes::E_DROPOUT:
				// at the moment the probability of neuron being dropped off the network is fixed to 0.5
				this->layers.push_back(std::make_shared<CDropoutLayer>(prev_layer, l_size, input_size, l_activation, 0.5));
				break;
			}

			prev_layer = l_size;
		}

		int output_neurons = classes == 2 ? 1 : classes;

		// add final layer (with different backprop implementation)
		this->layers.push_back(
			std::make_shared<CCrossEntropyLayer>(prev_layer, output_neurons, input_size, ActivationFunction::E_SIGMOID)
		);

		// use full batch training by default
		this->train_method = std::make_shared<CFullBatch>(42);
	}

	void CPerceptron::set_lr(double lr) {
		this->lr = lr;
	}

	void CPerceptron::set_lr_decay_state(bool state) {
		this->lr_decay = state;
	}

	void CPerceptron::set_train_method(std::shared_ptr<CTrainMethod> train_method) {
		this->train_method = train_method;
	}

	void CPerceptron::fit(const CMatrix &X, const CMatrix &y, int max_iter) {
		// train model
		double current_lr = this->lr; // make a copy in case of lr decay usage
		this->train_method->set_x(X);
		this->train_method->set_y(y);

		// split for the mini batches
		for (int i = 0; i < max_iter; ++i) {
			this->train_method->epoch([=](const CMatrix &_x, const CMatrix &_y) {
				this->forwardprop(_x, i);
				this->update_grads();
				this->backwardprop(_x, _y);
			});

			if (this->lr_decay) {
				this->lr = std::pow(current_lr, i);
			}

			if (i % 100 == 0) {
				char outp[100];
				auto& predicted = this->forwardprop(X);
				double cost = this->cost(y, predicted);

				sprintf_s(outp, "epoch: %d, cost: %lf, lr: %lf", i, cost, this->lr);
				std::cout << outp << std::endl;
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
			if(epoch != -1)
				layer->forward(prev_output, epoch + (int)time(nullptr) + clock());
			else
				layer->forward(prev_output, -1);
			
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