#pragma once
#include <vector>
#include <algorithm>

#include "Activation.h"
#include "Layer.h"

#include <Windows.h>
#define DBOUT( s )            \
{                             \
   std::ostringstream os_;    \
   os_ << s;                   \
   OutputDebugStringA( os_.str().c_str() );  \
}

namespace NN {
	class CPerceptron {
	private:
		std::vector<std::shared_ptr<CLayer>> layers;

		double lr = 0.35;
		bool lr_decay = false;
	public:
		class UnconvergentException {};

		CPerceptron(
			int input_size, 
			int vars,
			int classes,
			const std::vector<int>& hidden_layers, 
			const std::vector<ActivationFunction>& activators
		);

		void set_lr(double lr);
		void set_lr_decay_state(bool lr_decay);

		void fit(const CMatrix &X, const CMatrix &y, int max_iter = 5000);

		CMatrix predict(const CMatrix& x);

		CMatrix forwardprop(const CMatrix& x, int epoch = -1);

		void backwardprop(const CMatrix& x, const CMatrix& y);

		void update_grads();

		double cost(const CMatrix& real, const CMatrix& nn_out);
	};
}