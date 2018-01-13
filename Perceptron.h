#pragma once
#include <vector>
#include <algorithm>
#include <tuple>

#include "Activation.h"
#include "MiniBatch.h"
#include "Layer.h"

namespace NN {
	using layer_size = unsigned int;


	class CPerceptron {
	private:
		std::vector<std::shared_ptr<CLayer>> layers;
		std::shared_ptr<CTrainMethod> train_method;

		double lr = 0.35;
		bool lr_decay = false;
	public:
		class UnconvergentException {};

		CPerceptron(
			int input_size,
			int vars,
			int classes,
			std::vector<std::tuple<layer_size, ActivationFunction, LayerTypes>>& layer_params
		);

		void set_lr(double lr);
		void set_lr_decay_state(bool lr_decay);
		void set_train_method(std::shared_ptr<CTrainMethod> train_method);

		void fit(const CMatrix &X, const CMatrix &y, int max_iter = 5000);

		CMatrix predict(const CMatrix& x);

		CMatrix forwardprop(const CMatrix& x, int epoch = -1);

		void backwardprop(const CMatrix& x, const CMatrix& y);

		void update_grads();

		double cost(const CMatrix& real, const CMatrix& nn_out);
	};
}