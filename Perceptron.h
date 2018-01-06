#pragma once
#include <vector>
#include <algorithm>

#include "Activation.h"

#include <Windows.h>
#define DBOUT( s )            \
{                             \
   std::ostringstream os_;    \
   os_ << s;                   \
   OutputDebugStringA( os_.str().c_str() );  \
}

namespace NN {
	class CLayer {
	public:
		std::shared_ptr<IActivation> activation_func;
		CMatrix A, Z, W, B;
		CMatrix dA, dZ, dW, dB;

	public:
		
		CLayer(int neurons_prev, int neurons, int observations, ActivationFunction activation) {
			if(activation == ActivationFunction::E_SIGMOID)
				activation_func = std::make_shared<NN::CSigmoid>();
			else if(activation == ActivationFunction::E_TANH)
				activation_func = std::make_shared<NN::CTanh>();
			else if (activation == ActivationFunction::E_RELU)
				activation_func = std::make_shared<NN::CReLu>();

			this->W = CMatrix(neurons, neurons_prev);
			this->dW = CMatrix(neurons, neurons_prev);

			this->B = CMatrix(neurons, 1);
			this->dB = CMatrix(neurons, 1);

			this->Z = CMatrix(neurons, observations);
			this->dZ = CMatrix(neurons, observations);

			this->A = CMatrix(neurons, observations);
			this->dA = CMatrix(neurons, observations);
		}
	};

	class CPerceptron {
	private:
		std::vector<std::shared_ptr<CLayer>> layers;

		double lr = 0.05;
	public:
		class UnconvergentException {};

		CPerceptron(int input_size, int vars, const std::vector<int>& layer_sizes, const std::vector<ActivationFunction>& activators, double lr = .05);

		void fit(const CMatrix &X, const CMatrix &y, int max_iter = 5000);

		CMatrix predict(const CMatrix& x);

		CMatrix forwardprop(const CMatrix& x);

		void backwardprop(const CMatrix& x, const CMatrix& y);

		void update_grads();

		double cost(const CMatrix& real, const CMatrix& nn_out);
	};
}